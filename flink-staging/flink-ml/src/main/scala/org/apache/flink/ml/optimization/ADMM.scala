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

import breeze.linalg.DenseVector
import org.apache.flink.api.common.functions.RichMapFunction
import org.apache.flink.api.scala.DataSet
import org.apache.flink.configuration.Configuration
import org.apache.flink.ml.common.FlinkMLTools.ModuloKeyPartitioner
import org.apache.flink.ml.common._
import org.apache.flink.ml.math.DenseVector
import org.apache.flink.ml.math.{SparseVector, DenseVector}
import org.apache.flink.ml.optimization.IterativeSolver.{LearningRate, ConvergenceThreshold, Iterations}
import org.apache.flink.ml.optimization.Solver.{RegularizationConstant, LossFunction}
import breeze.linalg.{Vector => BreezeVector, DenseVector => BreezeDenseVector}
import org.apache.flink.ml.math.Breeze._

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

class ADMM extends IterativeSolver {

  case class ADMMWeights(xVector: WeightVector, uVector: WeightVector, zVector: WeightVector)

  import ADMM._
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

    val numberOfIterations: Int = parameters(Iterations)
    val lossFunction = parameters(LossFunction)
    val learningRate = parameters(LearningRate)
    val regularizationConstant = parameters(RegularizationConstant)


    // Check if the number of blocks/partitions has been specified
    val blocks = parameters.get(Blocks) match {
      case Some(value) => value
      case None => data.getParallelism
    }

    // Initialize weights
    val initialWeightsADMM: DataSet[ADMMWeights] = createInitialWeightsADMM(data)

    val blockedInputAndWeightsDS: DataSet[(Block[LabeledVector], ADMMWeights)] = FlinkMLTools
      .block(data, blocks, Some(ModuloKeyPartitioner))
      .crossWithTiny(initialWeightsADMM)
      .map(x => x)


    val resultingDataAndWeights = blockedInputAndWeightsDS.iterate(numberOfIterations) {
      blockedInputAndWeightsDS => {

        // Update values for uVector and xVector
        val updatedLocalWeights = blockedInputAndWeightsDS.map {
          blockedInputAndWeights => {
            val inputBlock: Block[LabeledVector] = blockedInputAndWeights._1
            val admmWeights: ADMMWeights = blockedInputAndWeights._2
            localUpdate(admmWeights, inputBlock)
          }
        }

        // Sum zVector
        // TODO: Zvector needs to be scaled
        val updatedZvector = updatedLocalWeights.reduce{
          (left, right) => {
            val weightSum = left.xVector.weights.asBreeze + left.uVector.weights.asBreeze
            val interceptSum = left.xVector.intercept + left.uVector.intercept
            val sumWv = WeightVector(weightSum.fromBreeze, interceptSum)
            ADMMWeights(left.xVector, left.uVector, sumWv)}
        }

        // Broadcast the updated value of zVector
        val updatedWeights = updatedLocalWeights.crossWithTiny(updatedZvector).map{
          weightsTuple => {
            val weightsOldZ = weightsTuple._1
            val weightsNewZ = weightsTuple._2
            weightsOldZ.copy(zVector = weightsNewZ.zVector)
          }
        }

        // Replace old weights with new weights at each block
        blockedInputAndWeightsDS
          .crossWithTiny(updatedWeights)
          .map{dataAndWeights => {
            val dataBlock = dataAndWeights._1._1
            val newWeights = dataAndWeights._2
            (dataBlock, newWeights)
          }
        }
      }
    }

    resultingDataAndWeights.first(1).map(x => x._2.xVector)
  }

  private def localUpdate(admmWeights: ADMMWeights, inputBlock: Block[LabeledVector])
    : ADMMWeights = ???


  def createInitialWeightsADMM(data: DataSet[LabeledVector])
  : DataSet[ADMMWeights] = {

    // TODO: Faster way to do this?
    val dimensionDS = data.map(_.vector.size).reduce((a, b) => b)

    dimensionDS.map {
      dimension =>
        val values = Array.fill(dimension)(0.0)
        val wv = new WeightVector(DenseVector(values), 0.0)
        //TODO: Do I need to create a deep copy of wv here?
        ADMMWeights(wv, wv, wv)
    }
  }

  /** Sets the number of data blocks/partitions
    *
    * @param blocks
    * @return itself
    */
  def setBlocks(blocks: Int): this.type = {
    parameters.add(Blocks, blocks)
    this
  }


  /** Sets the number of local SDCA iterations
    *
    * @param localIterations
    * @return itselft
    */
  def setLocalIterations(localIterations: Int): this.type =  {
    parameters.add(LocalIterations, localIterations)
    this
  }
}

object ADMM {
  val WEIGHT_VECTOR ="weightVector"

  case object LocalIterations extends Parameter[Int] {
    val defaultValue = Some(10)
  }

  case object Blocks extends Parameter[Int] {
    val defaultValue: Option[Int] = None
  }
}
