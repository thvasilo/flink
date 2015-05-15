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

import org.apache.flink.ml.common.{LabeledVector, WeightVector, ParameterMap}
import org.apache.flink.ml.math.DenseVector
import org.apache.flink.ml.optimization._
import org.apache.flink.ml.regression.RegressionData._
import org.scalatest.{Matchers, FlatSpec}

import org.apache.flink.api.scala._
import org.apache.flink.test.util.FlinkTestBase


class GeneralizedLinearModelSuite extends FlatSpec with Matchers with FlinkTestBase {


  behavior of "The generalized linear model framework"

  it should "solve a simple linear regression" in {
    val env = ExecutionEnvironment.getExecutionEnvironment

    env.setParallelism(2)
    
    val lr = LinearRegression()
      .setRegularizationType(NoRegularization())
      .setIterations(800)
      .setSolverType("sgd")
      .setStepsize(1.0)


    val inputDS = env.fromCollection(data)
    val inputFeatures = inputDS.map(_.vector)
    lr.fit(inputDS)
    lr.predict(inputFeatures)

    val weightList: Seq[WeightVector] = lr.getWeightVectorDS.collect()

    weightList.size should equal(1)

    val weightVector: WeightVector = weightList.head

    val weights = weightVector.weights.asInstanceOf[DenseVector].data
    val weight0 = weightVector.intercept


    expectedWeights zip weights foreach {
      case (expectedWeight, weight) =>
        weight should be (expectedWeight +- 0.1)
    }
    weight0 should be (expectedWeight0 +- 0.1)
  }

}
