package cn.sspku.zx.ml.mlLearner;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.feature.Normalizer;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;

import scala.Tuple2;

public class NormalizerExample {

	public static void main(String[] args) {
		SparkConf conf = new SparkConf().setAppName("NormalizeExample");
		SparkContext sc = new SparkContext(conf);

		JavaRDD<LabeledPoint> data = MLUtils
				.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
				.toJavaRDD().cache();

		final Normalizer normalizer1 = new Normalizer();
		final Normalizer normalizer2 = new Normalizer(Double.MAX_VALUE);

		// Each sample in data1 will be normalized using $L^2$ norm.
		@SuppressWarnings("serial")
		JavaRDD<Tuple2<Double, Vector>> data1 = data
				.map(new Function<LabeledPoint, Tuple2<Double, Vector>>() {
					public Tuple2<Double, Vector> call(LabeledPoint x)
							throws Exception {
						return new Tuple2<Double, Vector>(x.label(),
								normalizer1.transform(x.features()));
					}
				});
		System.out.println(data1);

		// Each sample in data2 will be normalized using $L^\infty$ norm.
		@SuppressWarnings("serial")
		JavaPairRDD<Double, Vector> data2 = data
				.mapToPair(new PairFunction<LabeledPoint, Double, Vector>() {
					public Tuple2<Double, Vector> call(LabeledPoint x)
							throws Exception {
						return new Tuple2<Double, Vector>(x.label(),
								normalizer2.transform(x.features()));
					}
				});
		System.out.println(data2);

	}
}
