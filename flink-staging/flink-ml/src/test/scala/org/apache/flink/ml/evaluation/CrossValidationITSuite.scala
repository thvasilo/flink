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
package org.apache.flink.ml.evaluation

import org.apache.flink.api.scala._
import org.apache.flink.ml.common.ParameterMap
import org.apache.flink.ml.preprocessing.StandardScaler
import org.apache.flink.ml.regression.RegressionData._
import org.apache.flink.ml.regression.{MultipleLinearRegression, RegressionData}
import org.apache.flink.test.util.FlinkTestBase

import org.scalatest.{FlatSpec, Matchers}

class CrossValidationITSuite extends FlatSpec with Matchers with FlinkTestBase {
  behavior of "the cross-validation suite"

  it should "be able to split the input into K folds" in {
    // Original code from the Apache Spark project
    val env = ExecutionEnvironment.getExecutionEnvironment

    val data = env.fromCollection(1 to 100)
    val collectedData = data.collect().sorted

    val twoFolds = KFold(2).folds(data, 42L)
    twoFolds(0)._1.collect().sorted shouldEqual twoFolds(1)._2.collect().sorted
    twoFolds(0)._2.collect().sorted shouldEqual twoFolds(1)._1.collect().sorted

    for (folds <- 2 to 10) {
      for (seed <- 1 to 5) {
        val foldedDataSets = KFold(folds).folds(data, seed)
        foldedDataSets.length shouldEqual  folds

        foldedDataSets.foreach { case (training, testing) =>
          val result = testing.union(training).collect().sorted
          val testingSize = testing.collect().size.toDouble
          testingSize should be > 0.0

          // Within 4 standard deviations of the mean
          val p = 1 / folds.toDouble
          val range = 4 * math.sqrt(100 * p * (1 - p))
          val expected = 100 * p
          val lowerBound = expected - range
          val upperBound = expected + range
          //Ensure size of test data is within expected bounds
          testingSize should be > lowerBound
          testingSize should be < upperBound
          training.collect().size should be > 0

          // The combined set should contain all data
          result shouldEqual collectedData
        }
        // K fold cross validation should only have each element in the validation set exactly once
        foldedDataSets.map(_._2).reduce((x, y) => x.union(y)).collect().sorted shouldEqual
          data.collect().sorted
      }
    }
  }

  def fixture = new {
    val env = ExecutionEnvironment.getExecutionEnvironment

    import RegressionData._


    val inputDS = env.fromCollection(data)

    val mlr = MultipleLinearRegression()
      .setStepsize(10.0)
      .setIterations(100)

    println()
  }

  it should "compute the cross-validation score for a regressor" in {

    val f = fixture

    val cvScores = CrossValidation.crossValScore(f.mlr, f.inputDS, cv = KFold(5), seed = 0L)

    cvScores.map(ds => ds.collect().head).foreach(println)

  }

  it should "compute the cross-validation score for a custom scorer" in {
    val f = fixture

    val squaredLossScorer = new Scorer(RegressionScores.squaredLoss)

    val cvScore = CrossValidation.crossValScore(
      f.mlr, f.inputDS, scorerOption = Some(squaredLossScorer), cv = KFold(5), seed = 0L)

    cvScore.map(ds => ds.collect().head).foreach(println)

  }

  it should "be possible to run CV on a chained predictor" in {
    val f = fixture

    val scaler = StandardScaler()

    val chainedPredictor = scaler.chainPredictor(f.mlr)

    val cvScores = CrossValidation.crossValScore(
      chainedPredictor, f.inputDS, cv = KFold(5), seed = 0L)

    cvScores.map(ds => ds.collect().head).foreach(println)
  }
}
