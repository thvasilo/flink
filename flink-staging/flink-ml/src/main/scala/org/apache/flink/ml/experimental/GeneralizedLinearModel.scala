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

package org.apache.flink.ml.experimental

import org.apache.flink.api.common.functions.RichMapFunction
import org.apache.flink.api.scala._
import org.apache.flink.configuration.Configuration
import org.apache.flink.ml.common.{Parameter, WeightVector, ParameterMap, LabeledVector}
import org.apache.flink.ml.math._
import org.apache.flink.ml.optimization._
import org.apache.flink.ml.experimental.GeneralizedLinearModel._

// Should the initialWeightsOption be an argument or a class variable that gets initialized some other way?
abstract class GeneralizedLinearModel(initialWeightsOption: Option[DataSet[WeightVector]],
                                      solverOption: Option[Solver]) extends
Predictor[GeneralizedLinearModel] {

  protected var solver: Solver = solverOption.getOrElse(GradientDescent())

  protected var weightVectorDS: DataSet[WeightVector] = null // Do a getOrElse here as well?

  def getWeightVectorDS: DataSet[WeightVector] = weightVectorDS

  private def getInitWeights: Option[DataSet[WeightVector]] = initialWeightsOption

  // TODO(tvas): Should we keep this?
  protected def createInitialWeightsDS(initialWeights: Option[DataSet[WeightVector]],
                             input: DataSet[LabeledVector]):  DataSet[WeightVector] = {
    solver.createInitialWeightsDS(initialWeights, input)
  }

  def setSolverType(solverName: String): GeneralizedLinearModel = {
    // Preserve parameters when a new solver is set (?)
    val oldParams = solver.parameters
    solver = solverName match {
      case "sgd" => GradientDescent()
      case s => {
        println(s"Solver $s is not implemented, solver is still ${solver.toString}")
        println("Supported solvers: \"sgd\": Stochastic gradient descent")
        solver
      }
    }
    // We might run into problems with the parameters if we move from an IterativeSolver to a
    // regular Solver. i.e. How can we ensure compatibility between Solver parameters?
    // Simple solution is to disregard old parameters and instruct user to reset them.
    solver.parameters = oldParams
    this
  }

  def setRegularizationValueParameter(regParam: Double) : GeneralizedLinearModel = {
    solver.setRegularizationParameter(regParam)
    this
  }

  def setIterations(iterations: Int) : GeneralizedLinearModel = {
    solver match {
      case its: IterativeSolver => its.setIterations(iterations)
      //TODO(tvas): Put this warning in the logger, also have name of solver instead of toString?
      case s => println(s"Solver ${s.toString} is not an iterative solver.")
    }
    this
  }

  def setStepsize(stepsize: Double) : GeneralizedLinearModel = {
    solver match {
      case its: IterativeSolver => its.setStepsize(stepsize)
      case s => println(s"Solver ${s.toString} is not an iterative solver.")
    }
    this
  }

  // TODO(tvas): I would like these to return the run-time type of the GLM, how could I do that?
  // I guess using a TypeTag and return asInstance of implicitly[TypeTag[Self]] could work
  def setConvergenceThreshold(convergenceThreshold: Double) : GeneralizedLinearModel = {
    solver match {
      case its: IterativeSolver => its.setConvergenceThreshold(convergenceThreshold)
      case s => println(s"Solver ${s.toString} is not an iterative solver.")
    }
    this
  }

  // TODO(tvas): Provide type matched access to miniBatchFraction for SGD

}

object GeneralizedLinearModel {
  val WEIGHTVECTOR_BROADCAST = "weights_broadcast"

  //--------------------------------------------------------------------------------------------------
  //  Prediction-related functions
  //--------------------------------------------------------------------------------------------------

  protected def pointDecisionFunction[V <: Vector](features: V, weightVector: WeightVector)
  : Double = {
    // TODO: The point decision functions can be defined somewhere where they are accessible by
    // both optimization and GLM class
    BLAS.dot(features, weightVector.weights) + weightVector.intercept
  }

  implicit def glmPredictor[V <: Vector]
  = new PredictOperation[GeneralizedLinearModel, V, LabeledVector] {
    override def predict(
                          instance: GeneralizedLinearModel,
                          parameters: ParameterMap,
                          input: DataSet[V]): DataSet[LabeledVector] = {
      input.map(new LinearPrediction[V]).withBroadcastSet(instance.weightVectorDS,
        WEIGHTVECTOR_BROADCAST)
    }
  }

  private class LinearPrediction[V <: Vector] extends RichMapFunction[V, LabeledVector] {

    var weightVector: WeightVector = null


    @throws(classOf[Exception])
    override def open(configuration: Configuration): Unit = {
      val list = this.getRuntimeContext.
        getBroadcastVariable[WeightVector](WEIGHTVECTOR_BROADCAST)

      weightVector = list.get(0)
    }

    override def map(example: V): LabeledVector = {

      val prediction = pointDecisionFunction(example, weightVector)

      LabeledVector(prediction, example)
    }
  }

  //------------------------------------------------------------------------------------------------
  //  Fit-related functions
  //------------------------------------------------------------------------------------------------
  //

  implicit def glmEstimator = new FitOperation[GeneralizedLinearModel, LabeledVector] {
      override def fit(
                        instance: GeneralizedLinearModel,
                        parameters: ParameterMap,
                        input: DataSet[LabeledVector]): Unit = {
        val instanceSolver = instance.solver
        instance.weightVectorDS = instance.createInitialWeightsDS(instance.getInitWeights, input)
        instance.weightVectorDS = instanceSolver.optimize(input, Some(instance.weightVectorDS))
      }
    }
}

/** Trait for GLMs for which we can choose the type of regularization **/
trait RegularizationOption extends GeneralizedLinearModel{
  def setRegularizationType(regularizationType: Regularization): RegularizationOption = {
    solver.setRegularizationType(regularizationType)
    this
  }
}

class LinearRegression(initialWeightsOption: Option[DataSet[WeightVector]], solverOption: Option[Solver]) extends
GeneralizedLinearModel(initialWeightsOption, solverOption) with RegularizationOption{
  solver
    .setLossFunction(SquaredLoss())
}

object LinearRegression {
  def apply(): LinearRegression = {
    new LinearRegression(None, None)
  }
}

class RidgeRegression(initialWeightsOption: Option[DataSet[WeightVector]], solverOption: Option[Solver])
  extends GeneralizedLinearModel(initialWeightsOption, solverOption) {

  solver
    .setLossFunction(SquaredLoss())
    .setRegularizationType(L2Regularization())

}

object RidgeRegression {
  def apply(): RidgeRegression = {
    new RidgeRegression(None, None)
  }
}

class Lasso(initialWeightsOption: Option[DataSet[WeightVector]], solverOption: Option[Solver])
  extends GeneralizedLinearModel(initialWeightsOption, solverOption) {

  solver
    .setLossFunction(SquaredLoss())
    .setRegularizationType(L1Regularization())

  // TODO(tvas): Override setSolver to ensure that only L1-compatible Solvers can be used?
  // Or delegate the sanitization of parameters to the Solvers?

}

object Lasso {
  def apply(): Lasso = {
    new Lasso(None, None)
  }
}