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
import org.apache.flink.ml.common.{ParameterMap, LabeledVector}
import org.apache.flink.ml.math.{DenseVector => FlinkDenseVector}
import org.apache.flink.ml.math.Breeze._

import breeze.linalg.{axpy => brzAxpy, Vector => BV, DenseVector => BDV,
DenseMatrix => BDM, sum => Bsum, argmax => Bargmax, norm => Bnorm, *}
import org.apache.flink.ml.pipeline.{FitOperation, PredictOperation, Predictor}

import scala.util.Random


trait ANNClassifierHelper {

  protected val labelToIndex: Map[Double, Int]
  private val indexToLabel = labelToIndex.map(_.swap)
  private val labelCount = labelToIndex.size

  protected def  labeledPointToVectorPair(labeledPoint: LabeledVector) = {
    val output = Array.fill(labelCount){0.1}
    output(labelToIndex(labeledPoint.label)) = 0.9
    (labeledPoint.vector.asInstanceOf[FlinkDenseVector], FlinkDenseVector(output))
  }

  protected def outputToLabel(output: FlinkDenseVector): Double = {
    val index = Bargmax(output.asBreeze.toDenseVector)
    indexToLabel(index)
  }
}

class ANNClassifier(
    val annModel: ArtificialNeuralNetwork,
    val labelToIndex: Map[Double, Int])
  extends ANNClassifierHelper with Predictor[ANNClassifier] with Serializable{

  /**
   * Predict values for a single data point using the model trained.
   *
   * @param testData array representing a single data point
   * @return predicted category from the trained model
   */
  def predictVector(testData: FlinkDenseVector): Double = {
    val output = annModel.predictVector(testData)
    outputToLabel(output)
  }
}

object ANNClassifier {

  implicit def fitANNClassifier =
    new FitOperation[ANNClassifier, LabeledVector] {
      override def fit(
          instance: ANNClassifier,
          fitParameters: ParameterMap,
          input: DataSet[LabeledVector])
      : Unit = {
        //TODO: Cast to (DenseVector, DenseVector)??
        val annData = input.map(lp => instance.labeledPointToVectorPair(lp))
        instance.annModel.fit(annData, fitParameters)
      }
    }

  /**
   * Returns random weights for the ANN classifier with the given hidden layers
   * and data dimensionality, i.e. the weights for the following topology:
   * [numFeatures -: hiddenLayers :- numLabels]
   *
   * @param data RDD containing labeled points for training.
   * @param hiddenLayersTopology number of nodes per hidden layer, excluding the bias nodes.
   * @param seed
   * @return vector with random weights.
   */
  def randomWeights(data: DataSet[LabeledVector],
                    hiddenLayersTopology: Array[Int], seed: Int): FlinkDenseVector = {
    /* TODO: remove duplication - the same analysis will be done in ANNClassifier.run() */
    val labelCount = data.map( lp => Tuple1(lp.label)).distinct().map(x => x._1)
      .collect()
      .length
    val featureCount = data.first(1).collect().head.vector.size
    ArtificialNeuralNetwork.randomWeights(featureCount, labelCount, hiddenLayersTopology, seed)
  }

  /**
   * Returns random weights for the ANN classifier with the given hidden layers
   * and data dimensionality, i.e. the weights for the following topology:
   * [numFeatures -: hiddenLayers :- numLabels]
   *
   * @param data RDD containing labeled points for training.
   * @param hiddenLayersTopology number of nodes per hidden layer, excluding the bias nodes.
   * @return vector with random weights.
   */
  def randomWeights(data: DataSet[LabeledVector], hiddenLayersTopology: Array[Int])
  : FlinkDenseVector = {
    randomWeights(data, hiddenLayersTopology, Random.nextInt())
  }

  /** Predict values for DataSet[DenseVector]
    *
    * @return The predict operation for DataSet[DenseVector]:
    *         predict(instance, parameters, input) => DataSet[(DenseVector, DenseVector)]
    */
  implicit def predictLabeledVectors = {
    new PredictOperation[
      ANNClassifier,
      LabeledVector,
      (Double, Double)] {
      override def predict(
          instance: ANNClassifier,
          predictParameters: ParameterMap,
          input: DataSet[LabeledVector])
      : DataSet[(Double, Double)] = {
        instance.annModel.weightsOption match {
          case Some(weights) => {
            input.map(lv =>
              (lv.label, instance.predictVector(lv.vector.asInstanceOf[FlinkDenseVector])))
          }
          case None => {
            throw new RuntimeException("The ArtificialNeuralNetwork has not been fitted to the " +
              "data. This is necessary to learn the weight vector.")
          }
        }

      }
    }
  }
}
