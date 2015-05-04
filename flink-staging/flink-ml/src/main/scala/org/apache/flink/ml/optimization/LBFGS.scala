/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.flink.ml.optimization

//import org.apache.flink.api.common.typeinfo.TypeInformation

import org.apache.flink.api.common.functions.RichMapFunction
import org.apache.flink.api.scala._
import org.apache.flink.configuration.Configuration
import org.apache.flink.api.scala.DataSet

import org.apache.flink.ml.common.{Parameter, ParameterMap, WeightVector, LabeledVector}
import org.apache.flink.ml.optimization.IterativeSolver._
import org.apache.flink.ml.optimization.LBFGS.HistoryLength
import org.apache.flink.ml.optimization.Solver._
import org.apache.flink.ml.math._
import org.apache.flink.ml.math.Breeze._


import breeze.linalg.{DenseVector => BreezeDenseVector}
import breeze.optimize.{CachedDiffFunction, DiffFunction, LBFGS => BreezeLBFGS}


class LBFGS(runParameters: ParameterMap) extends IterativeSolver{

  var parameterMap: ParameterMap = parameters ++ runParameters

  //Setters for parameters
  def setHistoryLength(length: Int): LBFGS = {
    parameterMap.add(Iterations, length)
    this
  }

  /** Provides a solution for the given optimization problem
    *
    * @param data A Dataset of LabeledVector (input, output) pairs
    * @param initialWeights The initial weight that will be optimized
    * @return A Vector of weights optimized to the given problem
    */
  override def optimize(data: DataSet[LabeledVector],
                        initialWeights: Option[DataSet[WeightVector]]): DataSet[WeightVector] = {

    val maxIterations = parameterMap(Iterations)
    val initialWeightsDS = createInitialWeightsDS(initialWeights, data)

    val costFun = new CostFun(data, initialWeightsDS)

    // NOTE (tvas): By usin g Breeze's LBFGS the regularization is applied to the intercept as
    // well. We can mitigate this by introducing intercept scaling, like sklearn does.
    val lbfgs = new BreezeLBFGS[BreezeDenseVector[Double]](
      maxIterations,
      parameterMap(HistoryLength),
      parameterMap(ConvergenceThreshold))

    // Create a DataSet containing the last state from the LBFGS iterations
    val tpe = createTypeInformation[lbfgs.State]
    val lastStateDS: DataSet[lbfgs.State] = initialWeightsDS.map{
      initialWVector: WeightVector => {
        val states = lbfgs.iterations(
          new CachedDiffFunction(costFun),
          initialWVector.weights.asBreeze.toDenseVector)
        val lastState: lbfgs.State = states.drop(states.size - 1).next()
        lastState
      }}

//    // Get the last state from the iterator
//    val lastStateDS: DataSet[lbfgs.State] = statesDS.map(
//      (statesIter: Iterator[lbfgs.State]) => statesIter.drop(statesIter.size - 1).next())

    // We are assuming that the intercept is at the first element of the weight vector (i.e. the
    // data have been augmented by adding a 1.0 feature at the beginning of each feature vector)
    val weightsDS = lastStateDS.map(_.x).map((brzWeights: BreezeDenseVector[Double]) =>
      WeightVector(brzWeights(1 to (brzWeights.length - 1)).fromBreeze, brzWeights(0)))

    weightsDS

    // The following can be used instead of the sequence of maps over one item above, but it needs
    // a pipeline breaking collect.
//    val initWeightsVector = initialWeightsDS.collect().head.weights

//    val states =
//      lbfgs.iterations(new CachedDiffFunction(costFun), initWeightsVector.asBreeze.toDenseVector)

//    val lastState = states.drop(states.size - 1).next()

//        val weights = lastState.x.fromBreeze
//
//        val optimizedWeights =  WeightVector(weights, 0.0)
//
//        initialWeightsDS.map( weightVector => optimizedWeights)

  }

  /**
   * CostFun implements Breeze's DiffFunction[T], which returns the loss and gradient
   * at a particular point (weights). It's used in Breeze's convex optimization routines.
   */
  private class CostFun(data: DataSet[LabeledVector], currentWeightsDS: DataSet[WeightVector])
    extends DiffFunction[BreezeDenseVector[Double]] {

    override def calculate(weights: BreezeDenseVector[Double]):
    (Double, BreezeDenseVector[Double]) = {

      // TODO: This should be abstracted higher up, as a calculateGradient(weights, lossFunction,
      // regularizationType) function
      val (gradientWV, lossSum, numExamples): (WeightVector, Double, Int) = data.map {
        new GradientLossCalculation
      }.withBroadcastSet(currentWeightsDS, WEIGHTVECTOR_BROADCAST).reduce {
        (left, right) =>
          val (leftGradVector, leftLoss, leftCount) = left
          val (rightGradVector, rightLoss, rightCount) = right

          BLAS.axpy(1.0, leftGradVector.weights, rightGradVector.weights)
          val gradients = WeightVector(
            rightGradVector.weights, leftGradVector.intercept + rightGradVector.intercept)

          (gradients, leftLoss + rightLoss, leftCount + rightCount)
      }.collect().head
      // TODO(tvas): Avoid the collect here

      val loss = lossSum / numExamples

      // gradientTotal = gradientSum / numExamples + gradientTotal
      BLAS.axpy(1.0 / numExamples, gradientWV.weights, gradientWV.weights)

      (loss, gradientWV.weights.asBreeze.asInstanceOf[BreezeDenseVector[Double]])
    }
  }

  /** Mapping function that calculates the weight gradients from the data.
    *
    */
  private class GradientLossCalculation extends
  RichMapFunction[LabeledVector, (WeightVector, Double, Int)] {

    var weightVector: WeightVector = null

    @throws(classOf[Exception])
    override def open(configuration: Configuration): Unit = {
      val list = this.getRuntimeContext.
        getBroadcastVariable[WeightVector](WEIGHTVECTOR_BROADCAST)

      weightVector = list.get(0)
    }

    override def map(example: LabeledVector): (WeightVector, Double, Int) = {

      val lossFunction = parameterMap(LossFunction)
      // TODO(tvas): Only allow differentiable regularization type, throw IllegalArgumentException
      val regType = parameterMap(RegularizationType)
      val regParameter = parameterMap(RegularizationParameter)
      val dimensions = example.vector.size
      // TODO(tvas): Any point in carrying the weightGradient vector for in-place replacement?
      // The idea in spark is to avoid object creation, but here we have to do it anyway
      val weightGradient = new DenseVector(new Array[Double](dimensions))

      val (loss, lossDeriv) = lossFunction.lossAndGradient(example, weightVector, weightGradient,
        regType, regParameter)

      // Restrict the value of the loss derivative to avoid numerical instabilities
      val restrictedLossDeriv: Double = {
        if (lossDeriv < -IterativeSolver.MAX_DLOSS) {
          -IterativeSolver.MAX_DLOSS
        }
        else if (lossDeriv > IterativeSolver.MAX_DLOSS) {
          IterativeSolver.MAX_DLOSS
        }
        else {
          lossDeriv
        }
      }

      (new WeightVector(weightGradient, restrictedLossDeriv), loss, 1)
    }
  }

}

object LBFGS {

  // Define parameterMap for LBFGS
  case object HistoryLength extends Parameter[Int] {
    val defaultValue = Some(10)
  }

  def apply(): LBFGS = {
    new LBFGS(new ParameterMap())
  }

  def apply(parameterMap: ParameterMap): LBFGS = {
    new LBFGS(parameterMap)
  }
}




