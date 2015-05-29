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
import org.apache.flink.api.scala.DataSet
import org.apache.flink.api.scala._
import org.apache.flink.configuration.Configuration

import org.apache.flink.ml._
import org.apache.flink.ml.common.{Parameter, WeightVector, LabeledVector}
import org.apache.flink.ml.math.{BLAS, Vector => FlinkVector, DenseVector => FlinkDenseVector}
import org.apache.flink.ml.math.Breeze._

import org.apache.flink.ml.optimization.IterativeSolver._
import org.apache.flink.ml.optimization.LBFGS._
import org.apache.flink.ml.optimization.Solver._

import breeze.linalg.{DenseVector => BreezeDenseVector}
import breeze.optimize.{CachedDiffFunction, DiffFunction, LBFGS => BreezeLBFGS}

/** Solver that uses the quasi-Newton LBFGS algorithm
  *
  * The implementation is based on Apache Spark Mllib's LBFGS code:
  * http://git.io/vUgrB
  */
class LBFGS extends IterativeSolver {

  //Setters for parameters
  def setHistoryLength(length: Int): LBFGS = {
    parameters.add(Iterations, length)
    this
  }

  /** Provides a solution for the given optimization problem
    *
    * @param data A Dataset of LabeledVector (input, output) pairs
    * @param initialWeights The initial weight that will be optimized
    * @return A Vector of weights optimized to the given problem
    */
  override def optimize(
      data: DataSet[LabeledVector],
      initialWeights: Option[DataSet[WeightVector]])
    : DataSet[WeightVector] = {

    val maxIterations = parameters(Iterations)
    val lossFunction = parameters(LossFunction)

    // Initialize weights
    val initialWeightsDS: DataSet[WeightVector] = createInitialWeightsDS(initialWeights, data)

    val costFun = new CostFun(data, initialWeightsDS, lossFunction)

    // NOTE (tvas): By using Breeze's LBFGS the regularization is applied to the intercept as
    // well. We can mitigate this by introducing intercept scaling, like sklearn does.
    val lbfgs = new BreezeLBFGS[BreezeDenseVector[Double]](
      maxIterations,
      parameters(HistoryLength),
      parameters(ConvergenceThreshold))

    val initialWVector = initialWeightsDS.collect().head

    // Perform the iterations using Breeze's L-BFGS implementation
    val brzWeights = lbfgs.minimize(
      new CachedDiffFunction(costFun),
      initialWVector.weights.asBreeze.toDenseVector)

    // Create a DataSet containing the optimized weights, taken from the last state of the LBFGS
    // iterations
    //    val tpe = createTypeInformation[lbfgs.State] // Cannot determine the type, why?
    val weightsDS: DataSet[WeightVector] = initialWeightsDS.map{
      initialWVector: WeightVector => {
        // We are assuming the initial data was augmented with a bias term (1.0) at the first
        // element, so the intercept should be the first weight.
        WeightVector(
          brzWeights.fromBreeze[FlinkDenseVector],
          0.0)
      }}

    weightsDS
  }

  private class CostFun(
      data: DataSet[LabeledVector],
      initialWeightsDS: DataSet[WeightVector],
      lossFunction: LossFunction)
    extends DiffFunction[BreezeDenseVector[Double]] {

    override def calculate(weights: BreezeDenseVector[Double]):
    (Double, BreezeDenseVector[Double]) = {

      // Get the current weight vector that we will broadcast
      val currentWeightsDS = initialWeightsDS.map {
        initialWeights: WeightVector => WeightVector(weights.fromBreeze[FlinkVector], 0)
      }

      // TODO: This should be abstracted higher up, as a calculateGradient(weights, lossFunction,
      // regularizationType) function
      // TODO: This create a serialization error
//      val lossGradientDS = data.mapWithBcVariable(currentWeightsDS){
//        (dataPoint, weightVector) => {
//          val gradient = lossFunction.gradient(dataPoint, weightVector)
//          val loss = lossFunction.loss(dataPoint, weightVector)
//          (gradient, loss, 1)}
//      }.reduce {
//        (left, right) =>
//          val (leftGradVector, leftLoss, leftCount) = left
//          val (rightGradVector, rightLoss, rightCount) = right
//
//          BLAS.axpy(1.0, leftGradVector.weights, rightGradVector.weights)
//          val gradients = WeightVector(
//            rightGradVector.weights, leftGradVector.intercept + rightGradVector.intercept)
//
//          (gradients, leftLoss + rightLoss, leftCount + rightCount)
//      }.map{
//        gradientLossAndCount =>
//          val weightGradients = gradientLossAndCount._1
//          val lossSum = gradientLossAndCount._2
//          val count = gradientLossAndCount._3
//
//          BLAS.scal(1.0 / count, weightGradients.weights)
//
//          (lossSum / count, weightGradients.weights)
//      }
      // TODO: This is fine
      val lossGradientDS = data.map {
        new GradientCalculation
      }.withBroadcastSet(currentWeightsDS, WEIGHTVECTOR_BROADCAST).reduce {
        (left, right) =>
          val (leftGradVector, leftLoss, leftCount) = left
          val (rightGradVector, rightLoss, rightCount) = right

          BLAS.axpy(1.0, leftGradVector.weights, rightGradVector.weights)
          val gradients = WeightVector(
            rightGradVector.weights, leftGradVector.intercept + rightGradVector.intercept)

          (gradients, leftLoss + rightLoss, leftCount + rightCount)
      }.map {
        new LossGradientUpdate
      }.withBroadcastSet(currentWeightsDS, WEIGHTVECTOR_BROADCAST)

      // TODO(tvas): Any ideas on how to avoid the collect here?
      val (loss, gradient) = lossGradientDS.collect().head
      (loss, gradient.asBreeze.asInstanceOf[BreezeDenseVector[Double]])
    }
  }

  /** Mapping function that calculates the weight gradients from the data.
    *
    */
  protected class GradientCalculation
    extends RichMapFunction[LabeledVector, (WeightVector, Double, Int)] {

    var weightVector: WeightVector = null

    @throws(classOf[Exception])
    override def open(configuration: Configuration): Unit = {
      val list = this.getRuntimeContext.
        getBroadcastVariable[WeightVector](WEIGHTVECTOR_BROADCAST)

      weightVector = list.get(0)
    }

    override def map(example: LabeledVector): (WeightVector, Double, Int) = {

      val lossFunction = parameters(LossFunction)

      val gradient = lossFunction.gradient(example, weightVector)
      val loss = lossFunction.loss(example, weightVector)

      (gradient, loss, 1)
    }
  }

  /** Mapping function that calculates the regularized loss and gradient
    *
    */
  private class LossGradientUpdate extends
  RichMapFunction[(WeightVector, Double, Int), (Double, FlinkVector)] {

    var weightVector: WeightVector = null

    @throws(classOf[Exception])
    override def open(configuration: Configuration): Unit = {
      val list = this.getRuntimeContext.
        getBroadcastVariable[WeightVector](WEIGHTVECTOR_BROADCAST)

      weightVector = list.get(0)
    }

    override def map(gradientLossAndCount: (WeightVector, Double, Int)): (Double, FlinkVector) = {

      val weightGradients = gradientLossAndCount._1
      val lossSum = gradientLossAndCount._2
      val count = gradientLossAndCount._3

      BLAS.scal(1.0 / count, weightGradients.weights)
      (lossSum / count, weightGradients.weights)
    }
  }
}

object LBFGS {

  val WEIGHTVECTOR_BROADCAST = "weights_broadcast"

  // Define parameterMap for LBFGS
  case object HistoryLength extends Parameter[Int] {
    val defaultValue = Some(10)
  }

  def apply(): LBFGS = {
    new LBFGS()
  }
}
