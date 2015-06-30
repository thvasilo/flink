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

package org.apache.flink.ml.classification

import org.apache.flink.api.scala.DataSet
import org.apache.flink.ml.common.LabeledVector
import org.apache.flink.ml.evaluation.AccuracyScore
import org.apache.flink.ml.pipeline.{EvaluateDataSetOperation, Predictor}

trait Classifier[Self] extends Predictor[Self]{
  that: Self =>

  def score(testing: DataSet[LabeledVector])
           (implicit evaluateOperation: EvaluateDataSetOperation[Self, LabeledVector, Double]):
  DataSet[Double] = {
    new AccuracyScore().evaluate(this.evaluate[LabeledVector, Double](testing))
  }
}
