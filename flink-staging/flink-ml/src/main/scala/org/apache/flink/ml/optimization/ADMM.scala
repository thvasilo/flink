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

import org.apache.flink.api.common.functions.RichMapFunction
import org.apache.flink.api.java.aggregation.Aggregations
import org.apache.flink.api.scala._
import org.apache.flink.configuration.Configuration
import org.apache.flink.ml.common.FlinkMLTools.ModuloKeyPartitioner
import org.apache.flink.ml.common._
import org.apache.flink.ml._

import org.apache.flink.api.scala.DataSetUtils.utilsToDataSet

import org.apache.flink.ml.math.DenseVector
import org.apache.flink.ml.math.{SparseVector, DenseVector}
import org.apache.flink.ml.optimization.IterativeSolver.{LearningRate, ConvergenceThreshold, Iterations}
import org.apache.flink.ml.optimization.Solver.{RegularizationConstant, LossFunction}
import breeze.linalg.{Vector => BreezeVector, DenseVector => BreezeDenseVector}

import org.apache.flink.ml.math.Breeze._


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

    val blockedInputDS = FlinkMLTools
      .block(data, blocks, Some(ModuloKeyPartitioner))
      .zipWithUniqueId

    // TODO: Faster way to do this?
    val dimensionDS = data.map(_.vector.size).reduce((a, b) => b)

    val dimension = dimensionDS.collect().head

    val values = Array.fill(dimension)(0.0)
    val wv = new WeightVector(DenseVector(values), 0.0)
    //TODO: Do I need to create a deep copy of wv here?
    val initialAdmmWeights = ADMMWeights(wv, wv, wv)

    val blockedWeights = blockedInputDS.map(x => (x._1, initialAdmmWeights))

    val resultingWeights = blockedWeights.iterate(numberOfIterations) {
      blockedWeights => {
        // Update the weights locally
        val updatedLocalWeights = blockedWeights
          .coGroup(blockedInputDS).where(0).equalTo(0) {
          (weightsIt, dataIt) => {
            val weightsTuple = weightsIt.next()
            val id = weightsTuple._1
            val currentWeights = weightsTuple._2
            val dataBlock = dataIt.next()._2
            (id, localUpdate(currentWeights, dataBlock))
          }
        }

        // Sum zVector
        // TODO: Zvector needs to be scaled
        val updatedZvector = updatedLocalWeights
//          .map(weightsWithID => weightsWithID._2) // Remove the ID
          .reduce {
          (leftTuple, rightTuple) => {
            // We ignore the ID, this way we save the map operation above
            val left = leftTuple._2
            val right = rightTuple._2
            // Sum u + x for left and right and then sum them together
            val weightLeftSum = left.xVector.weights.asBreeze + left.uVector.weights.asBreeze
            val weightRightSum = right.xVector.weights.asBreeze + right.uVector.weights.asBreeze
            val weightSum = weightLeftSum + weightRightSum

            val interceptLeftSum = left.xVector.intercept + left.uVector.intercept
            val interceptRightSum = right.xVector.intercept + right.uVector.intercept
            val interceptSum = interceptLeftSum + interceptRightSum

            val sumVector = WeightVector(weightSum.fromBreeze, interceptSum)
            (0L, left.copy(zVector = sumVector))}
        }

        // Broadcast the updated value of zVector
        val updatedWeights = updatedLocalWeights.mapWithBcVariable(updatedZvector) {
          (weightsWithOldZTuple, weightsWithNewZTuple ) => {
            val id = weightsWithOldZTuple._1
            (id, weightsWithOldZTuple._2.copy(zVector = weightsWithNewZTuple._2.zVector))
          }
        }

        updatedWeights
      }
    }

    resultingWeights.first(1).map(x => x._2.xVector)
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
