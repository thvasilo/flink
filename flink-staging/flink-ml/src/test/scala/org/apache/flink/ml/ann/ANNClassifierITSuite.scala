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

import org.apache.flink.ml.common.{WeightVector, LabeledVector}
import org.apache.flink.ml.math.{DenseVector => FlinkDV}
import org.apache.flink.ml.optimization._
import org.scalatest.{FlatSpec, Matchers}

import org.apache.flink.api.scala._
import org.apache.flink.test.util.FlinkTestBase

class ANNClassifierITSuite extends FlatSpec with Matchers with FlinkTestBase {

  behavior of "ANN classifier implementation"

  it should "solve the XOR problem using SGD" in {
    val env = ExecutionEnvironment.getExecutionEnvironment

    env.setParallelism(2)

    val lossFunction = GenericLossFunction(SquaredLoss, LinearPrediction)


    val inputs = Array[Array[Double]](
      Array[Double](0,0),
      Array[Double](0,1),
      Array[Double](1,0),
      Array[Double](1,1)
    )
    val outputs = Array[Double](0, 1, 1, 0)
    val inputLV = inputs.zip(outputs).map{ case(input, output) =>
      new LabeledVector(output, FlinkDV(input))}

    val inputDS = env.fromCollection(inputLV)
    val labelToIndex = inputDS
      .map( lp => lp.label)
      .map( lp => Tuple1(lp)).distinct()
      .map(x => x._1)
      .collect().sorted.zipWithIndex.toMap

    val topology = Array[Int](2,5,2)

    val annLoss = new ANNLeastSquaresLossFunction(topology , batchSize = 1)

    val sgd = SimpleGradientDescent()
      .setStepsize(100.0)
      .setIterations(2000)
      .setLossFunction(annLoss)
//      .setConvergenceThreshold(1e-4)

    val lbfgs = LBFGS()
      .setStepsize(1.0)
      .setIterations(200)
      .setLossFunction(annLoss)
      .setConvergenceThreshold(1e-4)

    val optimizer = sgd

    val annModel = new ArtificialNeuralNetwork(optimizer, topology)
    val ann = new ANNClassifier(annModel, labelToIndex)
    ann.fit(inputDS)

    val weightVector = ann.annModel.weightsOption.getOrElse(WeightVector(FlinkDV(0.0), 0.0))

    println(weightVector.weights)

    val predictionPairs: DataSet[(Double, Double)] = ann.predict(inputDS)

    val predSeq = predictionPairs.collect()

    println(predSeq)

    predSeq.foreach{
      case (truth, prediction) => truth should be (prediction +- 0.01)
    }

  }
}
