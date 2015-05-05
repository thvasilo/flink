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

import org.apache.flink.ml.common.{LabeledVector, WeightVector, ParameterMap}
import org.apache.flink.ml.math.{BLAS, Vector => FlinkVector, DenseVector}
import org.scalatest.{Matchers, FlatSpec}

import org.apache.flink.api.scala._
import org.apache.flink.test.util.FlinkTestBase


class LossFunctionITSuite extends FlatSpec with Matchers with FlinkTestBase {

  behavior of "The optimization Loss Function implementations"

  it should "calculate squared loss correctly" in {
    val env = ExecutionEnvironment.getExecutionEnvironment

    env.setParallelism(2)

    val squaredLoss = new SquaredLoss

    val example = LabeledVector(1.0, DenseVector(2))
    val weightVector = new WeightVector(DenseVector(1.0), 1.0)
    val gradient = DenseVector(0.0)

    def linearPrediction: (FlinkVector, WeightVector) => Double = {
      (features: FlinkVector, weights: WeightVector) =>
        BLAS.dot(features, weights.weights) + weights.intercept
    }

    val (loss, lossDerivative) = squaredLoss.lossAndGradient(example, weightVector, gradient, new
        NoRegularization, 0.0, linearPrediction)

    loss should be (2.0 +- 0.001)

    lossDerivative should be (2.0 +- 0.001)

    gradient.data(0) should be (4.0 +- 0.001)

  }

}