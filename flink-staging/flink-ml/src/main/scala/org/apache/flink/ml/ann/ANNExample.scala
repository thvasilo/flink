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
package org.apache.flink.ml.ann

import org.apache.flink.api.scala._
import org.apache.flink.ml.common.LabeledVector
import org.apache.flink.ml.math.{DenseVector => FlinkDV}
import org.apache.flink.ml.optimization._


object ANNExample {

  def main (args: Array[String]) {
    var trainPath = ""
    var testPath = ""
    var outputPath = ""

    def parseParameters(args: Array[String]): Boolean = {
      if (args.length > 0) {
        if (args.length == 3) {
          trainPath = args(0)
          testPath = args(1)
          outputPath = args(2)
          true
        } else {
          System.err.println("Usage: ANNExample <train path> <test path> <result path>")
          false
        }
      } else {
        System.out.println("Fail. Usage: ANNExample <train path> <test path> <result path>")
        true
      }
    }

    if (!parseParameters(args)) {
      return
    }

    val env = ExecutionEnvironment.getExecutionEnvironment

    def readCSV(env: ExecutionEnvironment, path: String) = {
      env.readTextFile(path)
        .map(_.split(',').map(_.toDouble))
        .map(ar => LabeledVector(ar(0), FlinkDV(ar slice (1, ar.length))))
    }

    val train  = readCSV(env, trainPath)
    val test  = readCSV(env, testPath)

    //    val trainFeatureCount = train.first(1).collect().head.vector.size
    //    val testFeatureCount = test.first(1).collect().head.vector.size
    //
    //    println(s"train: $trainFeatureCount, test: $testFeatureCount")

    val labelToIndex = train
      .map( lp => lp.label)
      .map( lp => Tuple1(lp)).distinct()
      .map(x => x._1)
      .collect().sorted.zipWithIndex.toMap

    val topology = Array[Int](784,300,10)

    val annLoss = new ANNLeastSquaresLossFunction(topology , batchSize = 100)

    val sgd = SimpleGradientDescent()
      .setStepsize(0.03)
      .setIterations(1000)
      .setLossFunction(annLoss)
    //      .setConvergenceThreshold(1e-4)

    val lbfgs = LBFGS()
      .setIterations(40)
      .setLossFunction(annLoss)
      .setConvergenceThreshold(1e-4)

    val optimizer = sgd

    val annModel = new ArtificialNeuralNetwork(optimizer, topology)
    val ann = new ANNClassifier(annModel, labelToIndex)
    ann.fit(train)

//    val weightVector = ann.annModel.weightsOption.getOrElse(WeightVector(FlinkDV(0.0), 0.0))
//
//    println(weightVector.weights)

    val predictionPairs: DataSet[(Double, Double)] = ann.predict(test)

    val ts = (System.currentTimeMillis / 1000).toString

    predictionPairs.writeAsText(outputPath + "-" + ts)

//    val predSeq = predictionPairs.collect()
  }
}
