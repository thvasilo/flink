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

import breeze.linalg.{axpy => brzAxpy, Vector => BV, DenseVector => BDV,
DenseMatrix => BDM, sum => Bsum, argmax => Bargmax, norm => Bnorm, *}
import breeze.numerics.{sigmoid => Bsigmoid}
import org.apache.flink.api.scala._
import org.apache.flink.ml.common.{Parameter, ParameterMap, WeightVector, LabeledVector}
import org.apache.flink.ml.math.{DenseVector => FlinkDenseVector}
import org.apache.flink.ml.optimization._
import org.apache.flink.ml.math.Breeze._
import org.apache.flink.ml.pipeline.{PredictOperation, FitOperation, Predictor}

/**
 * Performs the training of an Artificial Neural Network (ANN)
 *
 * @param topology A vector containing the number of nodes per layer in the network, including
 * the nodes in the input and output layer, but excluding the bias nodes.
 */
class ArtificialNeuralNetwork(
    val optimizer: IterativeSolver,
    val topology: Array[Int],
    val useSeed: Boolean = false,
    val seed: Int = 0,
    val batchSize: Int = 1)
  extends Predictor[ArtificialNeuralNetwork] with NeuralHelper with Serializable{

  import ArtificialNeuralNetwork._

  // Stores the weights of the linear model after the fitting phase
  var weightsOption: Option[WeightVector] = None

  /**
   * Predicts values for a single data point using the trained model.
   *
   * @param testData represents a single data point.
   * @return prediction using the trained model.
   */
  def predictVector(testData: FlinkDenseVector): FlinkDenseVector = {
    FlinkDenseVector(computeValues(testData, topology.length - 1))
  }

  private def computeValues(testData: FlinkDenseVector, layer: Int): Array[Double] = {
    require(layer >=0 && layer < topology.length)
    /* TODO: BDM */
    this.weightsOption match {
      case Some(weightVector) => {
        val (weightMatrices, bias) = unrollWeights(
          weightVector.weights.asInstanceOf[FlinkDenseVector])
        val outputs = forwardRun(
          testData.asBreeze.toDenseVector.toDenseMatrix.t,
          weightMatrices,
          bias)
        outputs(layer).toArray
      }
      case None => {
        throw new RuntimeException("The ArtificialNeuralNetwork has not been fitted to the " +
          "data. This is necessary to learn the weight vector.")
      }
    }
  }

  /**
   * Returns output values of a given layer for a single data point using the trained model.
   *
   * @param testData RDD represents a single data point.
   * @param layer index of a network layer
   * @return output of a given layer.
   */
  def output(testData: FlinkDenseVector, layer: Int): FlinkDenseVector = {
    FlinkDenseVector(computeValues(testData, layer))
  }

  /**
   * Returns weights for a given layer in vector form.
   *
   * @param index index of a layer: ranges from 1 until topology.length.
   *              (no weights for the 0 layer)
   * @return weights.
   */
  def weightsByLayer(index: Int): FlinkDenseVector = {
    require(index > 0 && index < topology.length)
    this.weightsOption match {
      case Some(weightVector) => {
        val (weightMatrices, bias) = unrollWeights(
          weightVector.weights.asInstanceOf[FlinkDenseVector])
        val layerWeight = BDV.vertcat(
          weightMatrices(index).toDenseVector,
          bias(index).toDenseVector)
        FlinkDenseVector(layerWeight.toArray)
      }
      case None => {
        throw new RuntimeException("The ArtificialNeuralNetwork has not been fitted to the " +
          "data. This is necessary to learn the weight vector of the linear function.")
      }
    }
  }

  def setIterations(iterations: Int): this.type = {
    optimizer.setIterations(iterations)
    this
  }

  def setStepsize(stepsize: Double): this.type = {
    optimizer.setStepsize(stepsize)
    this
  }

  def setConvergenceThreshold(convergenceThreshold: Double): this.type = {
    optimizer.setConvergenceThreshold(convergenceThreshold)
    this
  }

  def setBatchsize(batchsize: Int): this.type = {
    parameters.add(Batchsize, batchsize)
    this
  }

  def useSeed(useSeed: Boolean): this.type = {
    parameters.add(UseSeed, useSeed)
    this
  }

  def Seed(seed: Int): this.type = {
    parameters.add(ANNSeed, seed)
    this
  }
}

object ArtificialNeuralNetwork {

  case object Batchsize extends Parameter[Int] {
    val defaultValue = Some(1)
  }

  case object UseSeed extends Parameter[Boolean] {
    val defaultValue = Some(false)
  }

  case object ANNSeed extends Parameter[Int] {
    val defaultValue = Some(0)
  }

/** Predict values for DataSet[DenseVector]
    *
    * @return The predict operation for DataSet[DenseVector]:
    *         predict(instance, parameters, input) => DataSet[(DenseVector, DenseVector)]
    */
  implicit def predictVectors = {
    new PredictOperation[
      ArtificialNeuralNetwork,
      FlinkDenseVector,
      (FlinkDenseVector, FlinkDenseVector)] {
      override def predict(
          instance: ArtificialNeuralNetwork,
          predictParameters: ParameterMap,
          input: DataSet[FlinkDenseVector])
        : DataSet[(FlinkDenseVector, FlinkDenseVector)] = {
        instance.weightsOption match {
          case Some(weights) => {
          input.map(t => (t, instance.predictVector(t)))
        }
          case None => {
            throw new RuntimeException("The ArtificialNeuralNetwork has not been fitted to the " +
              "data. This is necessary to learn the weight vector.")
          }
        }

      }
    }
  }

  implicit val fitANN =
    new FitOperation[ArtificialNeuralNetwork, (FlinkDenseVector, FlinkDenseVector)] {
      override def fit(
          instance: ArtificialNeuralNetwork,
          fitParameters: ParameterMap,
          input: DataSet[(FlinkDenseVector, FlinkDenseVector)])
        : Unit = {

//        val map = instance.parameters ++ fitParameters
        val batchSize = instance.parameters(Batchsize)
        val topology = instance.topology
        val useSeed = instance.parameters(UseSeed)
        val seed = instance.parameters(ANNSeed)

        // We split the data differently, depending on the size of the batch we use for training
        val data = if (batchSize == 1) {
          input.map(v =>
            LabeledVector(0.0, BDV.vertcat(
              v._1.asBreeze.toDenseVector,
              v._2.asBreeze.toDenseVector).fromBreeze[FlinkDenseVector]))
        } else {
          input.mapPartition { it =>
            it.grouped(batchSize).map { seq =>
              val size = seq.size
              val bigVector = new Array[Double](topology(0) * size + topology.last * size)
              var i = 0
              seq.foreach { case (in, out) =>
                System.arraycopy(in.toArray, 0, bigVector, i * topology(0), topology(0))
                System.arraycopy(out.toArray, 0, bigVector,
                  topology(0) * size + i * topology.last, topology.last)
                i += 1
              }
              LabeledVector(0.0, FlinkDenseVector(bigVector))
            }
          }
        }
        val initialRandomWeights = WeightVector(
          ArtificialNeuralNetwork.randomWeights(topology, useSeed, seed),
          0.0)
        val initialWeightsDS = instance.optimizer
          .createInitialWeightsDS(None, data)
          .map(wv => initialRandomWeights)
        val initialWSeq = initialWeightsDS.collect().head
        val optimizedWeights = instance
          .optimizer.optimize(data, Some(initialWeightsDS)).collect().head
        instance.weightsOption = Some(optimizedWeights)
      }
    }

  /**
   * Provides a random weights vector.
   *
   * @param hiddenLayersTopology number of nodes per hidden layer, excluding the bias nodes.
   * @return random weights vector.
   */
  def randomWeights(trainingData: DataSet[(FlinkDenseVector,FlinkDenseVector)],
                    hiddenLayersTopology: Array[Int]): FlinkDenseVector = {
    val topology = convertTopology(trainingData, hiddenLayersTopology)
    randomWeights(topology, useSeed = false)
  }

  /**
   * Provides a random weights FlinkDenseVector, using given random seed.
   *
   * @param trainingData DataSet containing (input, output) pairs for later training.
   * @param hiddenLayersTopology number of nodes per hidden layer, excluding the bias nodes.
   * @param seed random generator seed.
   * @return random weights FlinkDenseVector.
   */
  def randomWeights(trainingData: DataSet[(FlinkDenseVector,FlinkDenseVector)],
                    hiddenLayersTopology: Array[Int],
                    seed: Int): FlinkDenseVector = {
    val topology = convertTopology(trainingData, hiddenLayersTopology)
    randomWeights(topology, useSeed = true, seed)
  }

  /**
   * Provides a random weights FlinkDenseVector, using given random seed.
   *
   * @param inputLayerSize size of input layer.
   * @param outputLayerSize size of output layer.
   * @param hiddenLayersTopology number of nodes per hidden layer, excluding the bias nodes.
   * @param seed random generator seed.
   * @return random weights FlinkDenseVector.
   */
  def randomWeights(inputLayerSize: Int,
                    outputLayerSize: Int,
                    hiddenLayersTopology: Array[Int],
                    seed: Int): FlinkDenseVector = {
    val topology = inputLayerSize +: hiddenLayersTopology :+ outputLayerSize
    randomWeights(topology, useSeed = true, seed)
  }

  def convertTopology(input: DataSet[(FlinkDenseVector,FlinkDenseVector)],
                              hiddenLayersTopology: Array[Int] ): Array[Int] = {
    val firstElt = input.first(1).collect().head
    firstElt._1.size +: hiddenLayersTopology :+ firstElt._2.size
  }

  private def randomWeights(
    topology: Array[Int],
    useSeed: Boolean,
    seed: Int = 0): FlinkDenseVector = {
    val rand: scala.util.Random =
      if(!useSeed) new scala.util.Random() else new scala.util.Random(seed)
    var i: Int = 0
    var l: Int = 0
    val noWeights = {
      var tmp = 0
      var i = 1
      while (i < topology.length) {
        tmp = tmp + topology(i) * (topology(i - 1) + 1)
        i += 1
      }
      tmp
    }
    val initialWeightsArr = new Array[Double](noWeights)
    var pos = 0
    l = 1
    while (l < topology.length) {
      i = 0
      while (i < (topology(l) * (topology(l - 1) + 1))) {
        initialWeightsArr(pos) = (rand.nextDouble * 4.8 - 2.4) / (topology(l - 1) + 1)
        pos += 1
        i += 1
      }
      l += 1
    }
    val fdv = FlinkDenseVector(initialWeightsArr)
    fdv
  }
}

/**
 * Trait for roll/unroll weights and forward/back propagation in neural network
 */
private[ann] trait NeuralHelper {
  protected val topology: Array[Int]
  protected val weightCount =
    (for(i <- 1 until topology.length) yield topology(i) * topology(i - 1)).sum +
      topology.sum - topology(0)

  protected def unrollWeights(
      weights: FlinkDenseVector): (Array[BDM[Double]],
      Array[BDV[Double]]) = {
    require(weights.size == weightCount)
    val weightsCopy = weights.data
    val weightMatrices = new Array[BDM[Double]](topology.length)
    val bias = new Array[BDV[Double]](topology.length)
    var offset = 0
    for(i <- 1 until topology.length){
      weightMatrices(i) = new BDM[Double](topology(i), topology(i - 1), weightsCopy, offset)
      offset += topology(i) * topology(i - 1)
      /* TODO: BDM */
      bias(i) = new BDV[Double](weightsCopy, offset, 1, topology(i))
      offset += topology(i)
    }
    (weightMatrices, bias)
  }

  protected def rollWeights(weightMatricesUpdate: Array[BDM[Double]],
                            biasUpdate: Array[BDV[Double]],
                            cumGradient: FlinkDenseVector): Unit = {
    val wu = cumGradient.data
    var offset = 0
    for(i <- 1 until topology.length){
      var k = 0
      val numElements = topology(i) * topology(i - 1)
      while(k < numElements){
        wu(offset + k) += weightMatricesUpdate(i).data(k)
        k += 1
      }
      offset += numElements
      k = 0
      while(k < topology(i)){
        wu(offset + k) += biasUpdate(i).data(k)
        k += 1
      }
      offset += topology(i)
    }
  }

  protected def forwardRun(data: BDM[Double], weightMatrices: Array[BDM[Double]],
                           bias: Array[BDV[Double]]): Array[BDM[Double]] = {
    val outArray = new Array[BDM[Double]](topology.length)
    outArray(0) = data
    for(i <- 1 until topology.length) {
      outArray(i) = weightMatrices(i) * outArray(i - 1)// :+ bias(i))
      outArray(i)(::, *) :+= bias(i)
      Bsigmoid.inPlace(outArray(i))
    }
    outArray
  }

  protected def wGradient(weightMatrices: Array[BDM[Double]],
                          targetOutput: BDM[Double],
                          outputs: Array[BDM[Double]]):
  (Array[BDM[Double]], Array[BDV[Double]]) = {
    /* error back propagation */
    val deltas = new Array[BDM[Double]](topology.length)
    val avgDeltas = new Array[BDV[Double]](topology.length)
    for(i <- (topology.length - 1) until (0, -1)){
      /* TODO: GEMM? */
      val outPrime = BDM.ones[Double](outputs(i).rows, outputs(i).cols)
      outPrime :-= outputs(i)
      outPrime :*= outputs(i)
      if(i == topology.length - 1){
        deltas(i) = (outputs(i) :- targetOutput) :* outPrime
      }else{
        deltas(i) = (weightMatrices(i + 1).t * deltas(i + 1)) :* outPrime
      }
      avgDeltas(i) = Bsum(deltas(i)(*, ::))
      avgDeltas(i) :/= outputs(i).cols.toDouble
    }
    /* gradient */
    val gradientMatrices = new Array[BDM[Double]](topology.length)
    for(i <- (topology.length - 1) until (0, -1)) {
      /* TODO: GEMM? */
      gradientMatrices(i) = deltas(i) * outputs(i - 1).t
      /* NB! dividing by the number of instances in
       * the batch to be transparent for the optimizer */
      gradientMatrices(i) :/= outputs(i).cols.toDouble
    }
    (gradientMatrices, avgDeltas)
  }
}

private class ANNLeastSquaresLossFunction(
    val topology: Array[Int],
    val batchSize: Int = 1) extends LossFunction with NeuralHelper {

  override def lossGradient(dataPoint: LabeledVector, weightVector: WeightVector)
    : (Double, WeightVector) = {
    val LabeledVector(_, vector) = dataPoint
    val weights = weightVector.weights.asInstanceOf[FlinkDenseVector]
    val arrData = vector.asInstanceOf[FlinkDenseVector].data
    val realBatchSize = arrData.length / (topology(0) + topology.last)
    val input = new BDM(topology(0), realBatchSize, arrData)
    val target = new BDM(topology.last, realBatchSize, arrData, topology(0) * realBatchSize)
    val (weightMatrices, bias) = unrollWeights(weights)
    /* forward run */
    val outputs = forwardRun(input, weightMatrices, bias)
    /* error back propagation */
    val cumGradient = FlinkDenseVector.zeros(weights.size)
    val (gradientMatrices, deltas) = wGradient(weightMatrices, target, outputs)
    rollWeights(gradientMatrices, deltas, cumGradient)
    /* error */
    val diff = target :- outputs(topology.length - 1)
    val outerError = Bsum(diff :* diff) / 2
    /* NB! dividing by the number of instances in
     * the batch to be transparent for the optimizer */
    val adjustedLoss = outerError / realBatchSize
    (adjustedLoss, WeightVector(cumGradient, 0.0))
  }
}
