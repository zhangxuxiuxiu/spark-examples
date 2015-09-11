package cn.sspku.zx.ml.mlLearner.classification;

import java.util.HashMap;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.util.MLUtils;

import scala.Tuple2;

public class DecisionTreeClassification {

	public static void main(String[] args) {
		SparkConf sparkConf = new SparkConf().setAppName("JavaDecisionTree");
		JavaSparkContext sc = new JavaSparkContext(sparkConf);

		// Load and parse the data file.
		// Cache the data since we will use it again to compute training error.
		String datapath = "data/mllib/sample_libsvm_data.txt";
		JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(sc.sc(), datapath)
				.toJavaRDD().cache();

		// Set parameters.
		// Empty categoricalFeaturesInfo indicates all features are continuous.
		Integer numClasses = 2;
		HashMap<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
		String impurity = "gini";
		Integer maxDepth = 5;
		Integer maxBins = 100;

		// Train a DecisionTree model for classification.
		final DecisionTreeModel model = DecisionTree.trainClassifier(data,
				numClasses, categoricalFeaturesInfo, impurity, maxDepth,
				maxBins);

		// Evaluate model on training instances and compute training error
		@SuppressWarnings("serial")
		JavaPairRDD<Double, Double> predictionAndLabel = data
				.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
					public Tuple2<Double, Double> call(LabeledPoint p)
							throws Exception {
						return new Tuple2<Double, Double>(model.predict(p
								.features()), p.label());
					}

				});
		@SuppressWarnings("serial")
		Double trainErr = 1.0
				* predictionAndLabel.filter(
						new Function<Tuple2<Double, Double>, Boolean>() {
							public Boolean call(Tuple2<Double, Double> pl)
									throws Exception {
								return !pl._1().equals(pl._2());
							}

						}).count() / data.count();
		System.out.println("Training error: " + trainErr);
		System.out.println("Learned classification tree model:\n" + model);
	}

}
