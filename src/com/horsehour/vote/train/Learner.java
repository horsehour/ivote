package com.horsehour.vote.train;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;
import java.util.stream.Collectors;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.lang3.tuple.Triple;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import com.horsehour.ml.classifier.LinearMachine;
import com.horsehour.ml.classifier.NeuralDecisionForest;
import com.horsehour.ml.classifier.NeuralDecisionForest.ErrorFunction;
import com.horsehour.ml.classifier.NeuralDecisionTree;
import com.horsehour.ml.classifier.tree.mtree.MDT;
import com.horsehour.ml.classifier.tree.mtree.MaximumCutPlane;
import com.horsehour.ml.classifier.tree.mtree.MaximumCutPlane.OptUpdateAlgo;
import com.horsehour.ml.data.Data;
import com.horsehour.ml.classifier.tree.mtree.MultiDecisionTree;
import com.horsehour.util.MathLib;
import com.horsehour.vote.ChoiceTriple;
import com.horsehour.vote.DataEngine;
import com.horsehour.vote.Profile;
import com.horsehour.vote.rule.LearnedRule;
import com.horsehour.vote.rule.VotingRule;
import com.horsehour.vote.train.Eval1.DataBridge;

import smile.classification.SVM.Multiclass;
import smile.math.Math;
import smile.math.distance.EuclideanDistance;
import smile.math.kernel.LinearKernel;

/**
 * Learn a voting rule satisfying many criteria, such as Condorcet winner
 * criterion, neutrality criterion, monotonicity criterion, based on certain
 * machine learning algorithm, e.g. decision tree, random forest, AdaBoost, LDA,
 * Perceptron, SVM, even a neutral network. The learned rule should be closely
 * approximated to the structure of the ground truth voting rule, and could
 * reach a compromise between different axiomatic properties.
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 2:50:51 PM, July 3, 2016
 */
public class Learner {
	public Function<Float, Float> logistic = x -> (float) (1.0 / (1.0 + Math.exp(-x)));
	public String meta = "";

	public Learner() {}

	public Learner(String meta) {
		this.meta = meta;
	}

	public VotingRule learn(List<ChoiceTriple<Integer>> profiles, String nameAlgo) {
		Pair<double[][], int[]> trainset = DataEngine.getFlatDataSet(profiles);
		int numItem = profiles.get(0).getProfile().getNumItem();

		Function<Profile<Integer>, List<Integer>> mechanism = null;

		if (nameAlgo.contains("knn")) {
			smile.math.distance.Distance<double[]> f = (x, y) -> MathLib.Distance.euclidean(x, y);
			smile.classification.KNN.Trainer<double[]> trainer = new smile.classification.KNN.Trainer<>(f, numItem);
			smile.classification.KNN<double[]> algo = trainer.train(trainset.getLeft(), trainset.getRight());

			mechanism = profile -> {
				List<Float> feature = DataEngine.getFeatures(profile);
				double[] x = new double[feature.size()];
				for (int i = 0; i < feature.size(); i++)
					x[i] = feature.get(i);
				return Arrays.asList(algo.predict(x));
			};
		} else if (nameAlgo.contains("lfd")) {
			smile.classification.FLD.Trainer trainer = new smile.classification.FLD.Trainer();
			smile.classification.FLD algo = trainer.train(trainset.getLeft(), trainset.getRight());

			mechanism = profile -> {
				List<Float> feature = DataEngine.getFeatures(profile);
				double[] x = new double[feature.size()];
				for (int i = 0; i < feature.size(); i++)
					x[i] = feature.get(i);
				return Arrays.asList(algo.predict(x));
			};
		} else if (nameAlgo.contains("lda")) {
			smile.classification.LDA.Trainer trainer = new smile.classification.LDA.Trainer();
			smile.classification.LDA algo = trainer.train(trainset.getLeft(), trainset.getRight());
			mechanism = profile -> {
				List<Float> feature = DataEngine.getFeatures(profile);
				double[] x = new double[feature.size()];
				for (int i = 0; i < feature.size(); i++)
					x[i] = feature.get(i);
				return Arrays.asList(algo.predict(x));
			};
		} else if (nameAlgo.contains("qda")) {
			smile.classification.QDA.Trainer trainer = new smile.classification.QDA.Trainer();
			smile.classification.QDA algo = trainer.train(trainset.getLeft(), trainset.getRight());
			mechanism = profile -> {
				List<Float> feature = DataEngine.getFeatures(profile);

				double[] x = new double[feature.size()];
				for (int i = 0; i < feature.size(); i++)
					x[i] = feature.get(i);
				return Arrays.asList(algo.predict(x));
			};
		} else if (nameAlgo.contains("rda")) {
			float alpha = 1.0F;
			smile.classification.RDA.Trainer trainer = new smile.classification.RDA.Trainer(alpha);
			smile.classification.RDA algo = trainer.train(trainset.getLeft(), trainset.getRight());

			mechanism = profile -> {
				List<Float> feature = DataEngine.getFeatures(profile);

				double[] x = new double[feature.size()];
				for (int i = 0; i < feature.size(); i++)
					x[i] = feature.get(i);
				return Arrays.asList(algo.predict(x));
			};
		} else if (nameAlgo.contains("adaboost")) {
			smile.classification.AdaBoost.Trainer trainer = new smile.classification.AdaBoost.Trainer();
			smile.classification.AdaBoost algo = trainer.train(trainset.getLeft(), trainset.getRight());
			mechanism = profile -> {
				List<Float> feature = DataEngine.getFeatures(profile);

				double[] x = new double[feature.size()];
				for (int i = 0; i < feature.size(); i++)
					x[i] = feature.get(i);
				return Arrays.asList(algo.predict(x));
			};
		} else if (nameAlgo.contains("logisticregression")) {
			smile.classification.LogisticRegression.Trainer trainer = null;
			trainer = new smile.classification.LogisticRegression.Trainer();
			smile.classification.LogisticRegression algo = trainer.train(trainset.getLeft(), trainset.getRight());
			mechanism = profile -> {
				List<Float> feature = DataEngine.getFeatures(profile);
				double[] x = new double[feature.size()];
				for (int i = 0; i < feature.size(); i++)
					x[i] = feature.get(i);
				return Arrays.asList(algo.predict(x));
			};
		} else if (nameAlgo.contains("naivebayes")) {
			int dim = trainset.getLeft()[0].length;

			smile.classification.NaiveBayes.Trainer trainer = new smile.classification.NaiveBayes.Trainer(smile.classification.NaiveBayes.Model.MULTINOMIAL, numItem, dim);
			smile.classification.NaiveBayes algo = trainer.train(trainset.getLeft(), trainset.getRight());

			mechanism = profile -> {
				List<Float> feature = DataEngine.getFeatures(profile);

				double[] x = new double[feature.size()];
				for (int i = 0; i < feature.size(); i++)
					x[i] = feature.get(i);
				return Arrays.asList(algo.predict(x));
			};
		} else if (nameAlgo.contains("svm")) {
			float c = 3;
			smile.classification.SVM.Trainer<double[]> trainer = new smile.classification.SVM.Trainer<>(new LinearKernel(), c, numItem, Multiclass.ONE_VS_ONE);
			smile.classification.SVM<double[]> algo = trainer.train(trainset.getLeft(), trainset.getRight());
			mechanism = profile -> {
				List<Float> feature = DataEngine.getFeatures(profile);
				double[] x = new double[feature.size()];
				for (int i = 0; i < feature.size(); i++)
					x[i] = feature.get(i);
				return Arrays.asList(algo.predict(x));
			};
		} else if (nameAlgo.contains("svmensemble")) {
			float c = 3;

			smile.classification.SVM.Trainer<double[]> trainer = null;

			/**
			 * using different kernels
			 */
			List<smile.math.kernel.MercerKernel<double[]>> kernelList = Arrays.asList(new smile.math.kernel.LinearKernel(), new smile.math.kernel.GaussianKernel(0.1), new smile.math.kernel.PolynomialKernel(2), new smile.math.kernel.HyperbolicTangentKernel(), new smile.math.kernel.HellingerKernel(), new smile.math.kernel.LaplacianKernel(0.2), new smile.math.kernel.PearsonKernel());

			List<smile.classification.SVM<double[]>> algoList = new ArrayList<>();

			int nKernel = kernelList.size();
			for (int i = 0; i < nKernel; i++) {
				trainer = new smile.classification.SVM.Trainer<double[]>(kernelList.get(i), c, numItem, Multiclass.ONE_VS_ONE);
				algoList.add(trainer.train(trainset.getLeft(), trainset.getRight()));
			}
			mechanism = profile -> {
				List<Float> feature = DataEngine.getFeatures(profile);
				double[] x = new double[feature.size()];
				for (int i = 0; i < feature.size(); i++)
					x[i] = feature.get(i);
				int[] preds = new int[nKernel];
				for (int k = 0; k < nKernel; k++)
					preds[k] = algoList.get(k).predict(x);
				int[] mode = MathLib.Data.mode(preds);
				List<Integer> winners = new ArrayList<>();
				for (int i = 0; i < mode.length; i++)
					winners.add(mode[i]);
				return winners;
			};
		} else if (nameAlgo.contains("rbfnetwork")) {
			smile.classification.RBFNetwork.Trainer<double[]> trainer = null;
			trainer = new smile.classification.RBFNetwork.Trainer<>(new EuclideanDistance());
			smile.classification.RBFNetwork<double[]> algo = trainer.train(trainset.getLeft(), trainset.getRight());

			mechanism = profile -> {
				List<Float> feature = DataEngine.getFeatures(profile);
				double[] x = new double[feature.size()];
				for (int i = 0; i < feature.size(); i++)
					x[i] = feature.get(i);
				return Arrays.asList(algo.predict(x));
			};
		} else if (nameAlgo.contains("decisiontree")) {
			int depth = 3, maxNode = 2 * depth - 1;
			smile.classification.DecisionTree.Trainer trainer = new smile.classification.DecisionTree.Trainer();
			trainer.setMaxNodes(maxNode);
			smile.classification.DecisionTree algo = trainer.train(trainset.getLeft(), trainset.getRight());
			mechanism = profile -> {
				List<Float> feature = DataEngine.getFeatures(profile);

				double[] x = new double[feature.size()];
				for (int i = 0; i < feature.size(); i++)
					x[i] = feature.get(i);
				return Arrays.asList(algo.predict(x));
			};
		} else if (nameAlgo.contains("scalarandomforest")) {
			SparkConf sparkConf = new SparkConf().setAppName("RandomForestClassifier").setMaster("local");
			JavaSparkContext jsc = new JavaSparkContext(sparkConf);

			JavaRDD<LabeledPoint> trainingData = DataBridge.getLabeledPoints(jsc, profiles);

			Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();

			String featureSubsetStrategy = "auto"; // Let the algorithm choose.
			String impurity = "entropy";
			int nClasse = profiles.get(0).getProfile().getNumItem(), nTree = 5, maxDepth = 5, maxBins = 32,
					seed = 12345;
			RandomForestModel model = org.apache.spark.mllib.tree.RandomForest.trainClassifier(trainingData, nClasse, categoricalFeaturesInfo, nTree, featureSubsetStrategy, impurity, maxDepth, maxBins, seed);

			mechanism = profile -> {
				List<Float> feature = DataEngine.getFeatures(profile);
				double[] vect = new double[feature.size()];
				for (int i = 0; i < feature.size(); i++)
					vect[i] = feature.get(i);
				Double pred = model.predict(Vectors.dense(vect));
				return Arrays.asList(pred.intValue());
			};
		} else if (nameAlgo.contains("mdt")) {
			MDT.Trainer trainer = new MDT.Trainer();
			int depth = 5;
			trainer.setMaxNodes(2 * depth - 1);
			trainer.width = numItem;
			trainer.numWidth = numItem;
			trainer.setSplitRule(MDT.SplitRule.CLASSIFICATION_ERROR);

			MDT algo = trainer.train(trainset.getLeft(), trainset.getRight());
			mechanism = profile -> {
				List<Float> feature = DataEngine.getFeatures(profile);
				double[] x = new double[feature.size()];
				for (int i = 0; i < feature.size(); i++)
					x[i] = feature.get(i);
				return Arrays.asList(algo.predict(x));
			};
		}

		LearnedRule learnedRule = new LearnedRule(mechanism);
		return learnedRule;
	}

	public VotingRule getMultiDecisionTreeRule(List<ChoiceTriple<Integer>> profiles, int depth) {
		Pair<double[][], int[]> trainset = DataEngine.getFlatDataSet(profiles);
		/**
		 * Training
		 */
		MultiDecisionTree.Trainer trainer = new MultiDecisionTree.Trainer();
		trainer.setMaxNodes(2 * depth - 1);
		trainer.setSplitRule(MultiDecisionTree.SplitRule.CLASSIFICATION_ERROR);

		MultiDecisionTree algo = trainer.train(trainset.getLeft(), trainset.getRight());

		/**
		 * Learned Rule (wrapped with a learned function / model)
		 */
		Function<Profile<Integer>, List<Integer>> mechanism = profile -> {
			List<Float> feature = DataEngine.getFeatures(profile);
			double[] x = new double[feature.size()];
			for (int i = 0; i < feature.size(); i++)
				x[i] = feature.get(i);
			return Arrays.asList(algo.predict(x));
		};
		LearnedRule learnedRule = new LearnedRule(mechanism);

		StringBuffer sb = new StringBuffer();
		sb.append(meta);
		sb.append("Multivariate Decision Tree (depth = " + depth + ")");
		learnedRule.setName(sb.toString());

		try {
			FileUtils.write(new File("csc/model.txt"), algo.toString(), "UTF8", true);
		} catch (IOException e) {
			e.printStackTrace();
		}
		return learnedRule;
	}

	/**
	 * Training based on Smile Random Forest algorithm
	 * 
	 * @param trainset
	 * @return learned voting rule
	 */
	public VotingRule getRandomForestRule(List<ChoiceTriple<Integer>> profiles) {
		Pair<double[][], int[]> trainset = DataEngine.getFlatDataSet(profiles);
		/**
		 * Training
		 */
		smile.classification.RandomForest.Trainer trainer = new smile.classification.RandomForest.Trainer();
		int depth = 5, nTree = 5;
		trainer.setMaxNodes(2 * depth - 1);
		trainer.setNumTrees(nTree);

		int dim = trainset.getLeft()[0].length;
		int nFeature = (int) Math.floor(Math.sqrt(dim));
		trainer.setNumRandomFeatures(nFeature);

		smile.classification.RandomForest algo = trainer.train(trainset.getLeft(), trainset.getRight());
		/**
		 * Learned Rule (wrapped with a learned function / model)
		 */
		Function<Profile<Integer>, List<Integer>> mechanism = profile -> {
			List<Float> feature = DataEngine.getFeatures(profile);

			double[] x = new double[feature.size()];
			for (int i = 0; i < feature.size(); i++)
				x[i] = feature.get(i);
			return Arrays.asList(algo.predict(x));
		};
		LearnedRule learnedRule = new LearnedRule(mechanism);

		StringBuffer sb = new StringBuffer();
		sb.append(meta);
		sb.append("Random Forest (nTree = " + nTree).append(", depth = " + depth).append(", nFeature = " + nFeature + ")");
		learnedRule.setName(sb.toString());

		try {
			FileUtils.write(new File("csc/model.txt"), algo.toString(), "UTF8", true);
		} catch (IOException e) {
			e.printStackTrace();
		}

		return learnedRule;
	}

	public List<LearnedRule> getDecomposedRandomForestRule(List<ChoiceTriple<Integer>> profiles) {
		Pair<double[][], int[]> trainset = DataEngine.getFlatDataSet(profiles);
		/**
		 * Training
		 */
		smile.classification.RandomForest.Trainer trainer = new smile.classification.RandomForest.Trainer();
		int depth = 5, nTree = 5;
		trainer.setMaxNodes(2 * depth - 1);
		trainer.setNumTrees(nTree);

		int dim = trainset.getLeft()[0].length;
		int nFeature = (int) Math.floor(Math.sqrt(dim));
		trainer.setNumRandomFeatures(nFeature);

		smile.classification.RandomForest algo = trainer.train(trainset.getLeft(), trainset.getRight());
		List<LearnedRule> rules = new ArrayList<>();
		/**
		 * Learned Rule (wrapped with a learned function / model)
		 */
		Function<Profile<Integer>, List<Integer>> mechanism = profile -> {
			List<Float> feature = DataEngine.getFeatures(profile);

			double[] x = new double[feature.size()];
			for (int i = 0; i < feature.size(); i++)
				x[i] = feature.get(i);
			return Arrays.asList(algo.predict(x));
		};
		rules.add(new LearnedRule(mechanism));

		StringBuffer sb = new StringBuffer();
		sb.append(meta);
		sb.append("Random Forest (nTree = " + nTree).append(", depth = " + depth).append(", nFeature = " + nFeature + ")");
		rules.get(rules.size() - 1).setName(sb.toString());

		try {
			FileUtils.write(new File("csc/model.txt"), algo.toString(), "UTF8", true);
		} catch (IOException e) {
			e.printStackTrace();
		}

		for (smile.classification.RandomForest.Tree tree : algo.trees) {
			Function<Profile<Integer>, List<Integer>> decisions = profile -> {
				List<Float> feature = DataEngine.getFeatures(profile);
				double[] x = new double[feature.size()];
				for (int i = 0; i < feature.size(); i++)
					x[i] = feature.get(i);
				return Arrays.asList(tree.tree.predict(x));
			};
			rules.add(new LearnedRule(decisions));
			rules.get(rules.size() - 1).setName("[Tree - Depth = " + depth + ", Weight = " + tree.weight + "]");
		}
		return rules;
	}

	/**
	 * Training based on Gradient Tree Boosting algorithm
	 * 
	 * @param trainset
	 * @return learned voting rule
	 */
	public VotingRule getGradientTreeBoostRule(List<ChoiceTriple<Integer>> profiles, int nTree, int depth) {
		Pair<double[][], int[]> trainset = DataEngine.getFlatDataSet(profiles);
		/**
		 * Training
		 */
		smile.classification.GradientTreeBoost.Trainer trainer = null;
		trainer = new smile.classification.GradientTreeBoost.Trainer(nTree);
		trainer.setMaxNodes(2 * depth - 1);

		smile.classification.GradientTreeBoost algo = trainer.train(trainset.getLeft(), trainset.getRight());
		/**
		 * Learned Rule (wrapped with a learned function / model)
		 */
		Function<Profile<Integer>, List<Integer>> mechanism = profile -> {
			List<Float> feature = DataEngine.getFeatures(profile);

			double[] x = new double[feature.size()];
			for (int i = 0; i < feature.size(); i++)
				x[i] = feature.get(i);
			return Arrays.asList(algo.predict(x));
		};
		LearnedRule learnedRule = new LearnedRule(mechanism);
		StringBuffer sb = new StringBuffer();
		sb.append(meta);
		sb.append("GDBT (nTree = " + nTree).append(", depth = " + depth + ")");
		learnedRule.setName(sb.toString());

		return learnedRule;
	}

	public VotingRule getNeuralNetworkRule(List<ChoiceTriple<Integer>> profiles) {
		Pair<double[][], int[]> trainset = DataEngine.getFlatDataSet(profiles);
		int dim = trainset.getLeft()[0].length, nItem = profiles.get(0).getProfile().getNumItem();

		int nHidden = 5;
		smile.classification.NeuralNetwork.Trainer trainer = null;
		trainer = new smile.classification.NeuralNetwork.Trainer(smile.classification.NeuralNetwork.ErrorFunction.CROSS_ENTROPY, dim, nHidden, nItem);
		smile.classification.NeuralNetwork algo = trainer.train(trainset.getLeft(), trainset.getRight());

		Function<Profile<Integer>, List<Integer>> mechanism = profile -> {
			List<Float> feature = DataEngine.getFeatures(profile);

			double[] x = new double[feature.size()];
			for (int i = 0; i < feature.size(); i++)
				x[i] = feature.get(i);
			return Arrays.asList(algo.predict(x));
		};
		LearnedRule learnedRule = new LearnedRule(mechanism);
		StringBuffer sb = new StringBuffer();
		sb.append(meta);
		sb.append("Neural Network (nhidden = " + nHidden + ", ErrorFunction = CrossEntropy)");
		learnedRule.setName(sb.toString());

		return learnedRule;
	}

	public VotingRule getDeepLearningRule(List<ChoiceTriple<Integer>> profiles) {
		Pair<double[][], int[]> trainset = DataEngine.getFlatDataSet(profiles);

		double[][] features = trainset.getLeft();
		int dim = features[0].length, len = features.length, nItem = profiles.get(0).getProfile().getNumItem();

		List<DataSet> dataset = new ArrayList<>();
		for (int i = 0; i < len; i++) {
			INDArray input = Nd4j.create(features[i]);
			INDArray output = Nd4j.zeros(1, nItem);
			output.putScalar(0, trainset.getRight()[i], 1);
			dataset.add(new DataSet(input, output));
		}

		int batch = dataset.size() / 20;
		DataSetIterator iter = new ListDataSetIterator(dataset, batch);

		int seed = 123;
		double learningRate = 0.001, momentum = 0.98;

		int nInput = dim, nOutput = nItem, nHidden = 20, nRun = 100;

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(seed).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.updater(Updater.NESTEROVS).iterations(1).activation("relu")
				.weightInit(WeightInit.XAVIER).learningRate(learningRate)
				.momentum(momentum).regularization(true)
				.l1(learningRate * 0.005).list()
				.layer(0, new DenseLayer.Builder().nIn(nInput).nOut(nHidden).build())
				.layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).activation("softmax").nIn(nHidden).nOut(nOutput).build())
				.pretrain(false).backprop(true).build();

		MultiLayerNetwork algo = new MultiLayerNetwork(conf);
		algo.init();
		// new HistogramIterationListener(1)
		algo.setListeners(new ScoreIterationListener(5));

		for (int i = 0; i < nRun; i++) {
			algo.fit(iter);
		}

		Function<Profile<Integer>, List<Integer>> mechanism = profile -> {
			List<Float> feature = DataEngine.getFeatures(profile);

			double[] x = new double[feature.size()];
			for (int i = 0; i < feature.size(); i++)
				x[i] = feature.get(i);
			int[] ret = algo.predict(Nd4j.create(x));
			return Arrays.asList(ret[0]);
		};

		LearnedRule learnedRule = new LearnedRule(mechanism);
		StringBuffer sb = new StringBuffer();
		sb.append(meta);
		sb.append("DL (").append("nhidden = " + nHidden).append(", lr = " + learningRate).append(", momentum = " + momentum).append(", optimizer = sgd").append(", activation = relu|softmax").append(", l1 = true").append(", lossfunction = negativeloglikelihood)");
		learnedRule.setName(sb.toString());

		try {
			FileUtils.write(new File("csc/model.txt"), algo.conf().toJson() + "\n", "UTF8", true);
		} catch (IOException e) {
			e.printStackTrace();
		}
		return learnedRule;
	}

	public VotingRule getCNNRule(List<ChoiceTriple<Integer>> profiles) {

		Pair<double[][], int[]> trainset = DataEngine.getFlatDataSet(profiles);

		double[][] features = trainset.getLeft();
		int len = features.length, nItem = profiles.get(0).getProfile().getNumItem();

		List<DataSet> dataset = new ArrayList<>();
		for (int i = 0; i < len; i++) {
			INDArray input = Nd4j.create(features[i]);
			INDArray output = Nd4j.zeros(1, nItem);
			output.putScalar(0, trainset.getRight()[i], 1);
			dataset.add(new DataSet(input, output));
		}

		int batch = 20;
		DataSetIterator iter = new ListDataSetIterator(dataset, batch);

		int seed = 123;
		double learningRate = 0.001;

		int nRun = 100, width = (int) Math.factorial(nItem);

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(Updater.NESTEROVS).iterations(1).activation("relu").weightInit(WeightInit.XAVIER).learningRate(learningRate).momentum(0.98).regularization(true).list().layer(0, new ConvolutionLayer.Builder(nItem, width).nOut(1).build()).layer(1, new OutputLayer.Builder(LossFunction.MCXENT).activation("softmax").nOut(nItem).build()).cnnInputSize(nItem, width, nItem).pretrain(false).backprop(true).build();

		MultiLayerNetwork algo = new MultiLayerNetwork(conf);
		algo.init();
		algo.setListeners(new ScoreIterationListener(5));

		for (int i = 0; i < nRun; i++) {
			algo.fit(iter);
		}

		Function<Profile<Integer>, List<Integer>> mechanism = profile -> {
			List<Float> feature = DataEngine.getFeatures(profile);

			double[] x = new double[feature.size()];
			for (int i = 0; i < feature.size(); i++)
				x[i] = feature.get(i);
			int[] ret = algo.predict(Nd4j.create(x));
			return Arrays.asList(ret[0]);
		};
		LearnedRule learnedRule = new LearnedRule(mechanism);

		StringBuffer sb = new StringBuffer();
		sb.append(meta);
		sb.append("CNN (lr = ").append(learningRate + ", ").append("optimizer = sgd, activation = relu|softmax)");
		learnedRule.setName(sb.toString());

		try {
			FileUtils.write(new File("csc/model.txt"), algo.conf().toJson() + "\n", "UTF8", true);
		} catch (IOException e) {
			e.printStackTrace();
		}
		return learnedRule;
	}

	public VotingRule getDLRFEnsembleRule(List<ChoiceTriple<Integer>> profiles) throws IOException {
		Pair<double[][], int[]> trainset = DataEngine.getFlatDataSet(profiles);
		double[][] features = trainset.getLeft();
		int dim = features[0].length, len = features.length, nItem = profiles.get(0).getProfile().getNumItem();

		List<DataSet> dataset = new ArrayList<>();
		for (int i = 0; i < len; i++) {
			INDArray input = Nd4j.create(features[i]);
			INDArray output = Nd4j.zeros(1, nItem);
			output.putScalar(0, trainset.getRight()[i], 1);
			dataset.add(new DataSet(input, output));
		}

		int batch = 20;
		DataSetIterator iter = new ListDataSetIterator(dataset, batch);

		/**
		 * multiple layer perceptron / neural network algorithm
		 */
		int seed = 123;
		double learningRate = 0.001;

		int nInput = dim, nOutput = nItem, nHidden = 20, nRun = 100;

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(Updater.NESTEROVS).iterations(1).activation("relu").weightInit(WeightInit.XAVIER).learningRate(learningRate).momentum(0.98).regularization(true).l1(learningRate * 0.005).list().layer(0, new DenseLayer.Builder().nIn(nInput).nOut(nHidden).build()).layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).activation("softmax").nIn(nHidden).nOut(nOutput).build()).pretrain(false).backprop(true).build();

		MultiLayerNetwork algoDL = new MultiLayerNetwork(conf);
		algoDL.init();
		// new HistogramIterationListener(1)
		algoDL.setListeners(new ScoreIterationListener(5));

		for (int i = 0; i < nRun; i++)
			algoDL.fit(iter);

		ModelSerializer.writeModel(algoDL, new File("csc/MLP.txt"), true);
		/**
		 * random forest algorithm
		 */
		smile.classification.RandomForest.Trainer trainer = new smile.classification.RandomForest.Trainer();
		int depth = 5, nTree = 5;
		trainer.setMaxNodes(2 * depth - 1);
		trainer.setNumTrees(nTree);

		int nFeature = (int) Math.floor(Math.sqrt(dim));
		trainer.setNumRandomFeatures(nFeature);

		smile.classification.RandomForest algoRF = trainer.train(trainset.getLeft(), trainset.getRight());
		/**
		 * Learned Rule (wrapped with a learned function / model)
		 */
		Function<Profile<Integer>, List<Integer>> mechanism = profile -> {
			List<Float> feature = DataEngine.getFeatures(profile);

			double[] x = new double[feature.size()];
			for (int i = 0; i < feature.size(); i++)
				x[i] = feature.get(i);

			double[] pred = new double[nItem];
			algoRF.predict(x, pred);

			/**
			 * importance coefficient to combine two models
			 */
			double lambda = 0.5;
			INDArray output = algoDL.output(Nd4j.create(x));
			for (int i = 0; i < nItem; i++)
				pred[i] = lambda * pred[i] + output.getDouble(0, i) * (1 - lambda);
			return Arrays.asList(Math.whichMax(pred));
		};
		LearnedRule learnedRule = new LearnedRule(mechanism);

		StringBuffer sb = new StringBuffer();
		sb.append(meta);
		sb.append("Ensemble - Random Forest and Deep Learning - ").append("SRF (nTree = " + nTree).append(", depth = " + depth + ", nFeature = " + nFeature + "), ").append("DL (nHidden = " + nHidden + ", ").append("lr = " + learningRate + ", ").append("batch = " + batch + ")");
		learnedRule.setName(sb.toString());

		try {
			FileUtils.write(new File("csc/model.txt"), algoDL.conf().toJson(), "UTF8", true);
		} catch (IOException e) {
			e.printStackTrace();
		}
		return learnedRule;
	}

	public VotingRule getPermutedNNRule(List<ChoiceTriple<Integer>> profiles) {
		Pair<double[][], int[]> trainset = DataEngine.getFlatDataSet(profiles);
		int dim = trainset.getLeft()[0].length, nItem = profiles.get(0).getProfile().getNumItem();

		List<Triple<Integer, Integer, Integer>> codeBook = new ArrayList<>();
		for (int i = 0; i < nItem; i++)
			for (int j = 0; j < nItem; j++)
				codeBook.add(Triple.of(i, -1, j));
		for (int i = 0; i < nItem; i++)
			for (int j = 0; j < nItem; j++)
				codeBook.add(Triple.of(i, j, i));

		List<List<Integer>> permutations = DataEngine.getAllPermutations(nItem);

		int nHidden = 10;
		smile.classification.NeuralNetwork.Trainer trainer = null;
		trainer = new smile.classification.NeuralNetwork.Trainer(smile.classification.NeuralNetwork.ErrorFunction.CROSS_ENTROPY, dim, nHidden, nItem);
		smile.classification.NeuralNetwork algo = trainer.train(trainset.getLeft(), trainset.getRight());

		int nPermutation = permutations.size();
		int[][] transfer = new int[nPermutation][dim];

		for (int k = 0; k < dim; k++)
			transfer[0][k] = k;// identity map

		for (int i = 1; i < nPermutation; i++) {
			List<Integer> permutation = permutations.get(i);
			for (int k = 0; k < dim; k++) {
				Triple<Integer, Integer, Integer> code = codeBook.get(k);
				int sid = code.getLeft(), pid = code.getMiddle(), pos = code.getRight();
				int psid = permutation.get(sid), ppid = -1, ppos = -1;
				if (pid == -1) {
					ppos = pos;
				} else {
					ppid = permutation.get(pid);
					ppos = psid;
				}

				int index = -1;
				for (Triple<Integer, Integer, Integer> codes : codeBook) {
					index++;
					if (codes.getLeft() == psid && codes.getMiddle() == ppid && codes.getRight() == ppos)
						break;
				}
				if (index == -1)
					continue;

				transfer[i][k] = index;
			}
		}

		Function<Profile<Integer>, List<Integer>> mechanism = profile -> {
			List<Float> feature = DataEngine.getFeatures(profile);
			double[] x = new double[dim];

			int[] pred = new int[nPermutation];
			for (int i = 0; i < nPermutation; i++) {
				List<Integer> permutation = permutations.get(i);
				for (int k = 0; k < dim; k++)
					x[k] = feature.get(transfer[i][k]);
				pred[i] = permutation.get(algo.predict(x));
			}

			List<Integer> winners = new ArrayList<>();
			int[] mode = MathLib.Data.mode(pred);
			for (int i = 0; i < mode.length; i++)
				winners.add(mode[i]);
			return winners;
		};

		LearnedRule learnedRule = new LearnedRule(mechanism);

		StringBuffer sb = new StringBuffer();
		sb.append(meta);
		sb.append("Permutated Neural Network (").append("nHidden = " + nHidden + ")");
		learnedRule.setName(sb.toString());

		return learnedRule;
	}

	public VotingRule getPermutedDLRule(List<ChoiceTriple<Integer>> profiles) {

		Pair<double[][], int[]> trainset = DataEngine.getFlatDataSet(profiles);

		double[][] features = trainset.getLeft();
		int dim = features[0].length, len = features.length, nItem = profiles.get(0).getProfile().getNumItem();

		List<DataSet> dataset = new ArrayList<>();
		for (int i = 0; i < len; i++) {
			INDArray input = Nd4j.create(features[i]);
			INDArray output = Nd4j.zeros(1, nItem);
			output.putScalar(0, trainset.getRight()[i], 1);
			dataset.add(new DataSet(input, output));
		}

		int batch = dataset.size() / 20;
		DataSetIterator iter = new ListDataSetIterator(dataset, batch);

		int seed = 123;
		double learningRate = 0.001;

		int nInput = dim, nOutput = nItem, nHidden = 20, nRun = 100;

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(Updater.NESTEROVS).iterations(1).activation("relu").weightInit(WeightInit.XAVIER).learningRate(learningRate).momentum(0.98).regularization(true).l1(learningRate * 0.005).list().layer(0, new DenseLayer.Builder().nIn(nInput).nOut(nHidden).build()).layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).activation("softmax").nIn(nHidden).nOut(nOutput).build()).pretrain(false).backprop(true).build();

		MultiLayerNetwork algo = new MultiLayerNetwork(conf);
		algo.init();
		// new HistogramIterationListener(1)
		algo.setListeners(new ScoreIterationListener(5));

		for (int i = 0; i < nRun; i++)
			algo.fit(iter);

		List<Triple<Integer, Integer, Integer>> codeBook = new ArrayList<>();
		for (int i = 0; i < nItem; i++)
			for (int j = 0; j < nItem; j++)
				codeBook.add(Triple.of(i, -1, j));
		for (int i = 0; i < nItem; i++)
			for (int j = 0; j < nItem; j++)
				codeBook.add(Triple.of(i, j, i));

		List<List<Integer>> permutations = DataEngine.getAllPermutations(nItem);
		int nPermutation = permutations.size();
		int[][] transfer = new int[nPermutation][dim];

		for (int k = 0; k < dim; k++)
			transfer[0][k] = k;// identity map

		for (int i = 1; i < nPermutation; i++) {
			List<Integer> permutation = permutations.get(i);
			for (int k = 0; k < dim; k++) {
				Triple<Integer, Integer, Integer> code = codeBook.get(k);
				int sid = code.getLeft(), pid = code.getMiddle(), pos = code.getRight();
				int psid = permutation.get(sid), ppid = -1, ppos = -1;
				if (pid == -1) {
					ppos = pos;
				} else {
					ppid = permutation.get(pid);
					ppos = psid;
				}

				int index = -1;
				for (Triple<Integer, Integer, Integer> codes : codeBook) {
					index++;
					if (codes.getLeft() == psid && codes.getMiddle() == ppid && codes.getRight() == ppos)
						break;
				}
				if (index == -1)
					continue;

				transfer[i][k] = index;
			}
		}

		Function<Profile<Integer>, List<Integer>> mechanism = profile -> {
			List<Float> feature = DataEngine.getFeatures(profile);

			int[] pred = new int[nPermutation];
			double[] x = new double[dim];

			for (int i = 0; i < nPermutation; i++) {
				List<Integer> permutation = permutations.get(i);
				for (int k = 0; k < dim; k++)
					x[k] = feature.get(transfer[i][k]);
				pred[i] = permutation.get(algo.predict(Nd4j.create(x))[0]);
			}

			List<Integer> winners = new ArrayList<>();
			int[] mode = MathLib.Data.mode(pred);
			for (int i = 0; i < mode.length; i++)
				winners.add(mode[i]);
			return winners;
		};
		LearnedRule learnedRule = new LearnedRule(mechanism);

		StringBuffer sb = new StringBuffer();
		sb.append(meta);
		sb.append("Permutated Deep Learning (").append("nHidden = " + nHidden + ", ").append("lr = " + learningRate + ", ").append("batch = " + batch + ")");
		learnedRule.setName(sb.toString());

		try {
			FileUtils.write(new File("csc/model.txt"), algo.conf().toJson() + "\n", "UTF8", true);
		} catch (IOException e) {
			e.printStackTrace();
		}
		return learnedRule;
	}

	/**
	 * Exercise Permutation on Decision Tree. Each node contains three
	 * permutable terms: self id (sid), pair id (pid), and its label (cid). Both
	 * self and pair ids are related to the split feature id. For positional
	 * feature, each fid is related to the corresponding candidate itself, and
	 * pid is non-permutable: pid = -1. For each pairwise feature, each fid
	 * connects with a candidate and its opponent. For non-determinal nodes, cid
	 * = -1. For leaf nodes, the label is permutated as the same as the other
	 * two terms.
	 * 
	 * @param profiles
	 *            positional and pairwise features
	 * @param depth
	 * @return learned voting rule
	 */
	public VotingRule getPermutedMDTRule(List<ChoiceTriple<Integer>> profiles, int depth) {
		Pair<double[][], int[]> trainset = DataEngine.getFlatDataSet(profiles);

		/**
		 * Training
		 */
		MultiDecisionTree.Trainer trainer = new MultiDecisionTree.Trainer();
		trainer.setMaxNodes(2 * depth - 1);
		trainer.setSplitRule(MultiDecisionTree.SplitRule.CLASSIFICATION_ERROR);

		MultiDecisionTree algo = trainer.train(trainset.getLeft(), trainset.getRight());

		int nItem = profiles.get(0).getProfile().getNumItem();
		List<Triple<Integer, Integer, Integer>> codeBook = new ArrayList<>();
		for (int i = 0; i < nItem; i++)
			for (int j = 0; j < nItem; j++)
				codeBook.add(Triple.of(i, -1, j));
		for (int i = 0; i < nItem; i++)
			for (int j = 0; j < nItem; j++)
				codeBook.add(Triple.of(i, j, i));

		List<List<Integer>> permutations = DataEngine.getAllPermutations(nItem);
		List<com.horsehour.ml.classifier.tree.mtree.MultiDecisionTree.Node> pdt = new ArrayList<>();
		int nPermutation = permutations.size();
		for (int i = 0; i < nPermutation; i++) {
			com.horsehour.ml.classifier.tree.mtree.MultiDecisionTree.Node root = null;
			if (i == 0)// identity map
				root = algo.root;
			else
				root = permuteNode(algo, algo.root, null, codeBook, permutations.get(i));
			pdt.add(root);
		}

		/**
		 * Learned Rule (wrapped with a learned function / model)
		 */
		Function<Profile<Integer>, List<Integer>> mechanism = profile -> {
			List<Float> feature = DataEngine.getFeatures(profile);
			double[] x = new double[feature.size()];
			for (int i = 0; i < feature.size(); i++)
				x[i] = feature.get(i);
			int[] pred = new int[nPermutation];
			for (int p = 0; p < nPermutation; p++) {
				pred[p] = pdt.get(p).predict(x);
			}
			int[] mode = MathLib.Data.mode(pred);
			List<Integer> winners = new ArrayList<>();
			for (int i = 0; i < mode.length; i++)
				winners.add(mode[i]);
			return winners;
		};
		LearnedRule learnedRule = new LearnedRule(mechanism);

		StringBuffer sb = new StringBuffer();
		sb.append(meta);
		sb.append("Permuted Multivariate Decision Tree (depth = " + depth + ")");
		learnedRule.setName(sb.toString());

		sb = new StringBuffer();
		for (int p = 0; p < nPermutation; p++)
			sb.append(pdt.get(p).toString() + "\n");

		try {
			FileUtils.write(new File("csc/model.txt"), sb.toString(), "UTF8", true);
		} catch (IOException e) {
			e.printStackTrace();
		}
		return learnedRule;
	}

	com.horsehour.ml.classifier.tree.mtree.MultiDecisionTree.Node permuteNode(
			com.horsehour.ml.classifier.tree.mtree.MultiDecisionTree algo,
			com.horsehour.ml.classifier.tree.mtree.MultiDecisionTree.Node parent,
			com.horsehour.ml.classifier.tree.mtree.MultiDecisionTree.Node child,
			List<Triple<Integer, Integer, Integer>> codeBook, List<Integer> permutation) {

		com.horsehour.ml.classifier.tree.mtree.MultiDecisionTree.Node nnode = algo.new Node();

		if (child == null) {
			int fid = parent.splitFeature;
			Triple<Integer, Integer, Integer> parentCode = codeBook.get(fid);
			int pid = parentCode.getMiddle();

			nnode.falseChild = permuteNode(algo, parent, parent.falseChild, codeBook, permutation);
			nnode.falseChildOutput = nnode.falseChild.output;

			nnode.trueChild = permuteNode(algo, parent, parent.trueChild, codeBook, permutation);
			nnode.trueChildOutput = nnode.trueChild.output;

			int sid = parentCode.getLeft();
			int psid = permutation.get(sid);
			int ppid = -1;
			if (pid > -1)
				ppid = permutation.get(pid);

			int index = -1;
			for (Triple<Integer, Integer, Integer> code : codeBook) {
				index++;
				if (code.getLeft() == psid && code.getMiddle() == ppid)
					break;
			}

			nnode.splitFeature = index;
			nnode.splitValue = parent.splitValue;
			return nnode;
		} else {
			if (child.falseChild != null) {
				nnode.falseChild = permuteNode(algo, child, child.falseChild, codeBook, permutation);
				nnode.falseChildOutput = child.falseChild.output;
			}

			if (child.trueChild != null) {
				nnode.trueChild = permuteNode(algo, child, child.trueChild, codeBook, permutation);
				nnode.trueChildOutput = child.trueChild.output;
			}

			int fid = parent.splitFeature;
			if (fid == -1 && parent.weightFeatures != null) {
				return null;
			} else {
				Triple<Integer, Integer, Integer> nodeCode = codeBook.get(fid);
				int pid = nodeCode.getMiddle();

				/**
				 * child is a leaf
				 */
				if (child.falseChild == null && child.trueChild == null) {
					nnode.output = child.output;
					if (pid > -1)
						nnode.output = permutation.get(child.output);
					return nnode;
				}

				fid = child.splitFeature;
				nodeCode = codeBook.get(fid);
				pid = nodeCode.getMiddle();

				int sid = nodeCode.getLeft();
				int psid = permutation.get(sid);
				int ppid = -1;
				if (pid > -1)
					ppid = permutation.get(pid);

				int index = -1;
				for (Triple<Integer, Integer, Integer> code : codeBook) {
					index++;
					if (code.getLeft() == psid && code.getMiddle() == ppid)
						break;
				}

				nnode.splitFeature = index;
				nnode.splitValue = child.splitValue;
				return nnode;
			}
		}
	}

	smile.classification.DecisionTree.Node permuteNode(smile.classification.DecisionTree algo,
			smile.classification.DecisionTree.Node parent, smile.classification.DecisionTree.Node child,
			List<Triple<Integer, Integer, Integer>> codeBook, List<Integer> permutation) {

		smile.classification.DecisionTree.Node nnode = algo.new Node();

		if (child == null) {
			int fid = parent.splitFeature;
			Triple<Integer, Integer, Integer> parentCode = codeBook.get(fid);
			int pid = parentCode.getMiddle();

			nnode.falseChild = permuteNode(algo, parent, parent.falseChild, codeBook, permutation);
			nnode.falseChildOutput = nnode.falseChild.output;

			nnode.trueChild = permuteNode(algo, parent, parent.trueChild, codeBook, permutation);
			nnode.trueChildOutput = nnode.trueChild.output;

			int sid = parentCode.getLeft();
			int psid = permutation.get(sid);
			int ppid = -1;
			if (pid > -1)
				ppid = permutation.get(pid);

			int index = -1;
			for (Triple<Integer, Integer, Integer> code : codeBook) {
				index++;
				if (code.getLeft() == psid && code.getMiddle() == ppid)
					break;
			}
			nnode.splitFeature = index;
			nnode.splitValue = parent.splitValue;
			return nnode;
		} else {
			if (child.falseChild != null) {
				nnode.falseChild = permuteNode(algo, child, child.falseChild, codeBook, permutation);
				nnode.falseChildOutput = child.falseChild.output;
			}

			if (child.trueChild != null) {
				nnode.trueChild = permuteNode(algo, child, child.trueChild, codeBook, permutation);
				nnode.trueChildOutput = child.trueChild.output;
			}

			int fid = parent.splitFeature;
			Triple<Integer, Integer, Integer> nodeCode = codeBook.get(fid);
			int pid = nodeCode.getMiddle();

			/**
			 * child is a leaf
			 */
			if (child.falseChild == null && child.trueChild == null) {
				nnode.output = child.output;
				if (pid > -1)
					nnode.output = permutation.get(child.output);
				return nnode;
			}

			fid = child.splitFeature;
			nodeCode = codeBook.get(fid);
			pid = nodeCode.getMiddle();

			int sid = nodeCode.getLeft();
			int psid = permutation.get(sid);
			int ppid = -1;
			if (pid > -1)
				ppid = permutation.get(pid);

			int index = -1;
			for (Triple<Integer, Integer, Integer> code : codeBook) {
				index++;
				if (code.getLeft() == psid && code.getMiddle() == ppid)
					break;
			}

			nnode.splitFeature = index;
			nnode.splitValue = child.splitValue;
			return nnode;
		}
	}

	public VotingRule getPermutedDTRule(List<ChoiceTriple<Integer>> profiles, int depth) {
		Pair<double[][], int[]> trainset = DataEngine.getFlatDataSet(profiles);
		double[][] features = trainset.getLeft();
		int dim = features[0].length, nItem = profiles.get(0).getProfile().getNumItem();

		List<Triple<Integer, Integer, Integer>> codeBook = new ArrayList<>();
		for (int i = 0; i < nItem; i++)
			for (int j = 0; j < nItem; j++)
				codeBook.add(Triple.of(i, -1, j));
		for (int i = 0; i < nItem; i++)
			for (int j = 0; j < nItem; j++)
				codeBook.add(Triple.of(i, j, i));

		List<List<Integer>> permutations = DataEngine.getAllPermutations(nItem);

		smile.classification.DecisionTree.Trainer trainerDT = null;
		trainerDT = new smile.classification.DecisionTree.Trainer();
		trainerDT.setMaxNodes(2 * depth - 1);
		trainerDT.setSplitRule(smile.classification.DecisionTree.SplitRule.CLASSIFICATION_ERROR);

		smile.classification.DecisionTree algoDT = trainerDT.train(trainset.getLeft(), trainset.getRight());

		List<smile.classification.DecisionTree.Node> pdt = new ArrayList<>();
		int nPermutation = permutations.size();
		for (int i = 0; i < nPermutation; i++) {
			smile.classification.DecisionTree.Node root = null;
			if (i == 0)// identity map
				root = algoDT.root;
			else
				root = permuteNode(algoDT, algoDT.root, null, codeBook, permutations.get(i));
			pdt.add(root);
		}

		int[][] transfer = new int[nPermutation][dim];

		for (int k = 0; k < dim; k++)
			transfer[0][k] = k;// identity map

		for (int i = 1; i < nPermutation; i++) {
			List<Integer> permutation = permutations.get(i);
			for (int k = 0; k < dim; k++) {
				Triple<Integer, Integer, Integer> code = codeBook.get(k);
				int sid = code.getLeft(), pid = code.getMiddle(), pos = code.getRight();
				int psid = permutation.get(sid), ppid = -1, ppos = -1;
				if (pid == -1) {
					ppos = pos;
				} else {
					ppid = permutation.get(pid);
					ppos = psid;
				}

				int index = -1;
				for (Triple<Integer, Integer, Integer> codes : codeBook) {
					index++;
					if (codes.getLeft() == psid && codes.getMiddle() == ppid && codes.getRight() == ppos)
						break;
				}
				if (index == -1)
					continue;

				transfer[i][k] = index;
			}
		}

		/**
		 * Learned Rule (wrapped with a learned function / model)
		 */
		Function<Profile<Integer>, List<Integer>> mechanism = profile -> {
			List<Float> feature = DataEngine.getFeatures(profile);
			double[] x = new double[feature.size()];
			for (int i = 0; i < feature.size(); i++)
				x[i] = feature.get(i);

			int[] pred = new int[nPermutation];
			for (int p = 0; p < nPermutation; p++)
				pred[p] = pdt.get(p).predict(x);

			List<Integer> winners = new ArrayList<>();
			int[] modes = MathLib.Data.mode(pred);
			for (int mode : modes)
				winners.add(mode);
			return winners;
		};
		LearnedRule learnedRule = new LearnedRule(mechanism);

		StringBuffer sb = new StringBuffer();
		sb.append(meta);
		sb.append("Permuted Decision Tree - ");
		sb.append("DT (depth = " + depth + ")");
		learnedRule.setName(sb.toString());

		sb = new StringBuffer();
		for (int i = 0; i < pdt.size(); i++)
			sb.append(pdt.get(i).toString() + "\n");

		try {
			FileUtils.write(new File("csc/model.txt"), sb.toString(), "UTF8", true);
		} catch (IOException e) {
			e.printStackTrace();
		}
		return learnedRule;
	}

	public VotingRule getPermutedMDTNNRule(List<ChoiceTriple<Integer>> profiles, int depth) {
		Pair<double[][], int[]> trainset = DataEngine.getFlatDataSet(profiles);
		double[][] features = trainset.getLeft();
		int dim = features[0].length, len = features.length, nItem = profiles.get(0).getProfile().getNumItem();

		List<DataSet> dataset = new ArrayList<>();
		for (int i = 0; i < len; i++) {
			INDArray input = Nd4j.create(features[i]);
			INDArray output = Nd4j.zeros(1, nItem);
			output.putScalar(0, trainset.getRight()[i], 1);
			dataset.add(new DataSet(input, output));
		}

		int batch = 20;
		DataSetIterator iter = new ListDataSetIterator(dataset, batch);

		/**
		 * multiple layer perceptron / neural network algorithm
		 */
		int seed = 123;
		double learningRate = 0.001;

		int nInput = dim, nOutput = nItem, nHidden = 20, nRun = 100;

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(Updater.NESTEROVS).iterations(1).activation("relu").weightInit(WeightInit.XAVIER).learningRate(learningRate).momentum(0.98).regularization(true).l1(learningRate * 0.005).list().layer(0, new DenseLayer.Builder().nIn(nInput).nOut(nHidden).build()).layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).activation("softmax").nIn(nHidden).nOut(nOutput).build()).pretrain(false).backprop(true).build();

		MultiLayerNetwork algoDL = new MultiLayerNetwork(conf);
		algoDL.init();
		// new HistogramIterationListener(1)
		algoDL.setListeners(new ScoreIterationListener(5));

		for (int i = 0; i < nRun; i++)
			algoDL.fit(iter);

		/**
		 * Training
		 */
		MultiDecisionTree.Trainer trainer = new MultiDecisionTree.Trainer();
		trainer.setMaxNodes(2 * depth - 1);
		trainer.setSplitRule(MultiDecisionTree.SplitRule.CLASSIFICATION_ERROR);

		MultiDecisionTree algo = trainer.train(trainset.getLeft(), trainset.getRight());

		List<Triple<Integer, Integer, Integer>> codeBook = new ArrayList<>();
		for (int i = 0; i < nItem; i++)
			for (int j = 0; j < nItem; j++)
				codeBook.add(Triple.of(i, -1, j));
		for (int i = 0; i < nItem; i++)
			for (int j = 0; j < nItem; j++)
				codeBook.add(Triple.of(i, j, i));

		List<List<Integer>> permutations = DataEngine.getAllPermutations(nItem);
		List<com.horsehour.ml.classifier.tree.mtree.MultiDecisionTree.Node> pdt = new ArrayList<>();
		int nPermutation = permutations.size();
		for (int i = 0; i < nPermutation; i++) {
			com.horsehour.ml.classifier.tree.mtree.MultiDecisionTree.Node root = null;
			if (i == 0)// identity map
				root = algo.root;
			else
				root = permuteNode(algo, algo.root, null, codeBook, permutations.get(i));
			pdt.add(root);
		}

		/**
		 * Learned Rule (wrapped with a learned function / model)
		 */
		Function<Profile<Integer>, List<Integer>> mechanism = profile -> {
			List<Float> feature = DataEngine.getFeatures(profile);
			double[] x = new double[feature.size()];
			for (int i = 0; i < feature.size(); i++)
				x[i] = feature.get(i);

			int[] pred = new int[nPermutation];
			for (int p = 0; p < nPermutation; p++)
				pred[p] = pdt.get(p).predict(x);

			List<Integer> winners = new ArrayList<>();
			int[] modes = MathLib.Data.mode(pred);
			for (int mode : modes)
				winners.add(mode);

			INDArray output = algoDL.output(Nd4j.create(x));

			double[] predNN = new double[nItem];
			for (int i = 0; i < nItem; i++) {
				predNN[i] = output.getDouble(0, i);
			}
			int index = Math.whichMax(predNN);
			if (!winners.contains(index))
				winners.add(index);
			return winners;
		};
		LearnedRule learnedRule = new LearnedRule(mechanism);

		StringBuffer sb = new StringBuffer();
		sb.append(meta);
		sb.append("Permuted Multivariate Decision Tree and Neural Netowrk - ").append("MDT (depth = " + depth).append("), NN (nhidden = " + nHidden + ", ").append("batch = " + batch + ")");
		learnedRule.setName(sb.toString());

		try {
			FileUtils.write(new File("csc/model.txt"), algoDL.toString(), "UTF8", true);
		} catch (IOException e) {
			e.printStackTrace();
		}
		return learnedRule;
	}

	public VotingRule getPermutedMDTPermutedNNRule(List<ChoiceTriple<Integer>> profiles, int depth) {
		Pair<double[][], int[]> trainset = DataEngine.getFlatDataSet(profiles);
		double[][] features = trainset.getLeft();
		int dim = features[0].length, len = features.length, nItem = profiles.get(0).getProfile().getNumItem();

		List<DataSet> dataset = new ArrayList<>();
		for (int i = 0; i < len; i++) {
			INDArray input = Nd4j.create(features[i]);
			INDArray output = Nd4j.zeros(1, nItem);
			output.putScalar(0, trainset.getRight()[i], 1);
			dataset.add(new DataSet(input, output));
		}

		int batch = 20;
		DataSetIterator iter = new ListDataSetIterator(dataset, batch);

		/**
		 * multiple layer perceptron / neural network algorithm
		 */
		int seed = 123;
		double learningRate = 0.001;

		int nInput = dim, nOutput = nItem, nHidden = 4, nRun = 100;

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(Updater.NESTEROVS).iterations(1).activation("relu").weightInit(WeightInit.XAVIER).learningRate(learningRate).momentum(0.98).regularization(true).l1(learningRate * 0.005).list().layer(0, new DenseLayer.Builder().nIn(nInput).nOut(nHidden).build()).layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).activation("softmax").nIn(nHidden).nOut(nOutput).build()).pretrain(false).backprop(true).build();

		MultiLayerNetwork algoNN = new MultiLayerNetwork(conf);
		algoNN.init();
		// new HistogramIterationListener(1)
		algoNN.setListeners(new ScoreIterationListener(5));

		for (int i = 0; i < nRun; i++)
			algoNN.fit(iter);

		/**
		 * Training
		 */
		MultiDecisionTree.Trainer trainer = new MultiDecisionTree.Trainer();
		trainer.setMaxNodes(2 * depth - 1);
		trainer.setSplitRule(MultiDecisionTree.SplitRule.CLASSIFICATION_ERROR);

		MultiDecisionTree algoMDT = trainer.train(trainset.getLeft(), trainset.getRight());

		List<Triple<Integer, Integer, Integer>> codeBook = new ArrayList<>();
		for (int i = 0; i < nItem; i++)
			for (int j = 0; j < nItem; j++)
				codeBook.add(Triple.of(i, -1, j));
		for (int i = 0; i < nItem; i++)
			for (int j = 0; j < nItem; j++)
				codeBook.add(Triple.of(i, j, i));

		List<List<Integer>> permutations = DataEngine.getAllPermutations(nItem);
		List<com.horsehour.ml.classifier.tree.mtree.MultiDecisionTree.Node> pdt = new ArrayList<>();
		int nPermutation = permutations.size();
		for (int i = 0; i < nPermutation; i++) {
			com.horsehour.ml.classifier.tree.mtree.MultiDecisionTree.Node root = null;
			if (i == 0)// identity map
				root = algoMDT.root;
			else
				root = permuteNode(algoMDT, algoMDT.root, null, codeBook, permutations.get(i));
			pdt.add(root);
		}

		int[][] transfer = new int[nPermutation][dim];

		for (int k = 0; k < dim; k++)
			transfer[0][k] = k;// identity map

		for (int i = 1; i < nPermutation; i++) {
			List<Integer> permutation = permutations.get(i);
			for (int k = 0; k < dim; k++) {
				Triple<Integer, Integer, Integer> code = codeBook.get(k);
				int sid = code.getLeft(), pid = code.getMiddle(), pos = code.getRight();
				int psid = permutation.get(sid), ppid = -1, ppos = -1;
				if (pid == -1) {
					ppos = pos;
				} else {
					ppid = permutation.get(pid);
					ppos = psid;
				}

				int index = -1;
				for (Triple<Integer, Integer, Integer> codes : codeBook) {
					index++;
					if (codes.getLeft() == psid && codes.getMiddle() == ppid && codes.getRight() == ppos)
						break;
				}
				if (index == -1)
					continue;

				transfer[i][k] = index;
			}
		}

		/**
		 * Learned Rule (wrapped with a learned function / model)
		 */
		Function<Profile<Integer>, List<Integer>> mechanism = profile -> {
			List<Float> feature = DataEngine.getFeatures(profile);
			double[] x = new double[feature.size()];
			for (int i = 0; i < feature.size(); i++)
				x[i] = feature.get(i);

			int[] pred = new int[nPermutation];
			for (int p = 0; p < nPermutation; p++)
				pred[p] = pdt.get(p).predict(x);

			List<Integer> winners = new ArrayList<>();
			int[] modes = MathLib.Data.mode(pred);
			for (int mode : modes)
				winners.add(mode);

			for (int i = 0; i < nPermutation; i++) {
				List<Integer> permutation = permutations.get(i);
				for (int k = 0; k < dim; k++)
					x[k] = feature.get(transfer[i][k]);
				pred[i] = algoNN.predict(Nd4j.create(x))[0];
				pred[i] = permutation.get(pred[i]);
			}

			int[] mode = MathLib.Data.mode(pred);
			for (int i = 0; i < mode.length; i++)
				if (!winners.contains(mode[i]))
					winners.add(mode[i]);
			return winners;
		};
		LearnedRule learnedRule = new LearnedRule(mechanism);

		StringBuffer sb = new StringBuffer();
		sb.append(meta);
		sb.append("Permuted Multivariate Decision Tree and Permuted Neural Netowrk - ");
		sb.append("MDT (depth = " + depth + "), NN (nhidden = " + nHidden + ")");
		learnedRule.setName(sb.toString());

		try {
			FileUtils.write(new File("csc/model.txt"), algoMDT.toString(), "UTF8", true);
		} catch (IOException e) {
			e.printStackTrace();
		}
		return learnedRule;
	}

	public VotingRule getPermutedDTPermutedNNRule(List<ChoiceTriple<Integer>> profiles, int depth) {
		Pair<double[][], int[]> trainset = DataEngine.getFlatDataSet(profiles);
		double[][] features = trainset.getLeft();
		int dim = features[0].length, nItem = profiles.get(0).getProfile().getNumItem();

		List<Triple<Integer, Integer, Integer>> codeBook = new ArrayList<>();
		for (int i = 0; i < nItem; i++)
			for (int j = 0; j < nItem; j++)
				codeBook.add(Triple.of(i, -1, j));
		for (int i = 0; i < nItem; i++)
			for (int j = 0; j < nItem; j++)
				codeBook.add(Triple.of(i, j, i));

		List<List<Integer>> permutations = DataEngine.getAllPermutations(nItem);

		smile.classification.DecisionTree.Trainer trainerDT = null;
		trainerDT = new smile.classification.DecisionTree.Trainer();
		trainerDT.setMaxNodes(2 * depth - 1);
		trainerDT.setSplitRule(smile.classification.DecisionTree.SplitRule.CLASSIFICATION_ERROR);

		smile.classification.DecisionTree algoDT = trainerDT.train(trainset.getLeft(), trainset.getRight());

		List<smile.classification.DecisionTree.Node> pdt = new ArrayList<>();
		int nPermutation = permutations.size();
		for (int i = 0; i < nPermutation; i++) {
			smile.classification.DecisionTree.Node root = null;
			if (i == 0)// identity map
				root = algoDT.root;
			else
				root = permuteNode(algoDT, algoDT.root, null, codeBook, permutations.get(i));
			pdt.add(root);
		}

		int[][] transfer = new int[nPermutation][dim];

		for (int k = 0; k < dim; k++)
			transfer[0][k] = k;// identity map

		for (int i = 1; i < nPermutation; i++) {
			List<Integer> permutation = permutations.get(i);
			for (int k = 0; k < dim; k++) {
				Triple<Integer, Integer, Integer> code = codeBook.get(k);
				int sid = code.getLeft(), pid = code.getMiddle(), pos = code.getRight();
				int psid = permutation.get(sid), ppid = -1, ppos = -1;
				if (pid == -1) {
					ppos = pos;
				} else {
					ppid = permutation.get(pid);
					ppos = psid;
				}

				int index = -1;
				for (Triple<Integer, Integer, Integer> codes : codeBook) {
					index++;
					if (codes.getLeft() == psid && codes.getMiddle() == ppid && codes.getRight() == ppos)
						break;
				}
				if (index == -1)
					continue;

				transfer[i][k] = index;
			}
		}

		int nHidden = 10;
		smile.classification.NeuralNetwork.Trainer trainerNN = null;
		trainerNN = new smile.classification.NeuralNetwork.Trainer(smile.classification.NeuralNetwork.ErrorFunction.CROSS_ENTROPY, dim, nHidden, nItem);

		smile.classification.NeuralNetwork algoNN = trainerNN.train(trainset.getLeft(), trainset.getRight());

		/**
		 * Learned Rule (wrapped with a learned function / model)
		 */
		Function<Profile<Integer>, List<Integer>> mechanism = profile -> {
			List<Float> feature = DataEngine.getFeatures(profile);
			double[] x = new double[feature.size()];
			for (int i = 0; i < feature.size(); i++)
				x[i] = feature.get(i);

			int[] pred = new int[nPermutation];
			for (int p = 0; p < nPermutation; p++)
				pred[p] = pdt.get(p).predict(x);

			List<Integer> winners = new ArrayList<>();
			int[] modes = MathLib.Data.mode(pred);
			for (int mode : modes)
				winners.add(mode);

			for (int i = 0; i < nPermutation; i++) {
				List<Integer> permutation = permutations.get(i);
				for (int k = 0; k < dim; k++)
					x[k] = feature.get(transfer[i][k]);
				pred[i] = permutation.get(algoNN.predict(x));
			}

			int[] mode = MathLib.Data.mode(pred);
			for (int i = 0; i < mode.length; i++)
				if (!winners.contains(mode[i]))
					winners.add(mode[i]);
			return winners;
		};
		LearnedRule learnedRule = new LearnedRule(mechanism);

		StringBuffer sb = new StringBuffer();
		sb.append(meta);
		sb.append("Permuted Decision Tree and Permuted Neural Netowrk - ");
		sb.append("DT (depth = " + depth + "), NN (nhidden = " + nHidden + ")");
		learnedRule.setName(sb.toString());

		sb = new StringBuffer();
		for (int i = 0; i < pdt.size(); i++)
			sb.append(pdt.get(i).toString() + "\n");

		try {
			FileUtils.write(new File("csc/model.txt"), sb.toString(), "UTF8", true);
		} catch (IOException e) {
			e.printStackTrace();
		}
		return learnedRule;
	}

	public VotingRule getPermutedLRRule(List<ChoiceTriple<Integer>> profiles) {
		Pair<double[][], int[]> trainset = DataEngine.getFlatDataSet(profiles);

		smile.classification.LogisticRegression.Trainer trainer = null;
		trainer = new smile.classification.LogisticRegression.Trainer();
		smile.classification.LogisticRegression algo = trainer.train(trainset.getLeft(), trainset.getRight());

		double[][] features = trainset.getLeft();
		int dim = features[0].length, nItem = profiles.get(0).getProfile().getNumItem();

		List<Triple<Integer, Integer, Integer>> codeBook = new ArrayList<>();
		for (int i = 0; i < nItem; i++)
			for (int j = 0; j < nItem; j++)
				codeBook.add(Triple.of(i, -1, j));
		for (int i = 0; i < nItem; i++)
			for (int j = 0; j < nItem; j++)
				codeBook.add(Triple.of(i, j, i));

		List<List<Integer>> permutations = DataEngine.getAllPermutations(nItem);
		int nPermutation = permutations.size();

		int[][] transfer = new int[nPermutation][dim];

		for (int k = 0; k < dim; k++)
			transfer[0][k] = k;// identity map

		for (int i = 1; i < nPermutation; i++) {
			List<Integer> permutation = permutations.get(i);
			for (int k = 0; k < dim; k++) {
				Triple<Integer, Integer, Integer> code = codeBook.get(k);
				int sid = code.getLeft(), pid = code.getMiddle(), pos = code.getRight();
				int psid = permutation.get(sid), ppid = -1, ppos = -1;
				if (pid == -1) {
					ppos = pos;
				} else {
					ppid = permutation.get(pid);
					ppos = psid;
				}

				int index = -1;
				for (Triple<Integer, Integer, Integer> codes : codeBook) {
					index++;
					if (codes.getLeft() == psid && codes.getMiddle() == ppid && codes.getRight() == ppos)
						break;
				}
				if (index == -1)
					continue;

				transfer[i][k] = index;
			}
		}

		/**
		 * Learned Rule (wrapped with a learned function / model)
		 */
		Function<Profile<Integer>, List<Integer>> mechanism = profile -> {
			List<Float> feature = DataEngine.getFeatures(profile);
			double[] x = new double[feature.size()];
			for (int i = 0; i < feature.size(); i++)
				x[i] = feature.get(i);

			int[] pred = new int[nPermutation];
			List<Integer> winners = new ArrayList<>();
			for (int i = 0; i < nPermutation; i++) {
				List<Integer> permutation = permutations.get(i);
				for (int k = 0; k < dim; k++)
					x[k] = feature.get(transfer[i][k]);
				pred[i] = permutation.get(algo.predict(x));
			}

			int[] mode = MathLib.Data.mode(pred);
			for (int i = 0; i < mode.length; i++)
				if (!winners.contains(mode[i]))
					winners.add(mode[i]);
			return winners;
		};

		LearnedRule learnedRule = new LearnedRule(mechanism);
		learnedRule.setName(meta + "Permuted Logistic Regression");
		return learnedRule;
	}

	/**
	 * Permutated Decision Tree and Logistic Regression Rule
	 * 
	 * @param profiles
	 * @return
	 */
	public VotingRule getPermutedDTPermutedLRRule(List<ChoiceTriple<Integer>> profiles, int depth) {
		Pair<double[][], int[]> trainset = DataEngine.getFlatDataSet(profiles);
		double[][] features = trainset.getLeft();
		int dim = features[0].length, nItem = profiles.get(0).getProfile().getNumItem();

		// <cand., p. cand., pos./lbl.>
		List<Triple<Integer, Integer, Integer>> codeBook = new ArrayList<>();
		for (int i = 0; i < nItem; i++)
			for (int j = 0; j < nItem; j++)
				codeBook.add(Triple.of(i, -1, j));
		for (int i = 0; i < nItem; i++)
			for (int j = 0; j < nItem; j++)
				codeBook.add(Triple.of(i, j, i));

		List<List<Integer>> permutations = DataEngine.getAllPermutations(nItem);

		smile.classification.DecisionTree.Trainer trainerDT = null;
		trainerDT = new smile.classification.DecisionTree.Trainer();
		trainerDT.setMaxNodes(2 * depth - 1);
		trainerDT.setSplitRule(smile.classification.DecisionTree.SplitRule.CLASSIFICATION_ERROR);

		smile.classification.DecisionTree algoDT = trainerDT.train(trainset.getLeft(), trainset.getRight());

		List<smile.classification.DecisionTree.Node> pdt = new ArrayList<>();
		int nPermutation = permutations.size();
		for (int i = 0; i < nPermutation; i++) {
			smile.classification.DecisionTree.Node root = null;
			if (i == 0)// identity map
				root = algoDT.root;
			else
				root = permuteNode(algoDT, algoDT.root, null, codeBook, permutations.get(i));
			pdt.add(root);
		}

		int[][] transfer = new int[nPermutation][dim];

		for (int k = 0; k < dim; k++)
			transfer[0][k] = k;// identity map, features are unchanged

		for (int i = 1; i < nPermutation; i++) {
			List<Integer> permutation = permutations.get(i);
			for (int k = 0; k < dim; k++) {
				Triple<Integer, Integer, Integer> code = codeBook.get(k);
				int sid = code.getLeft(), pid = code.getMiddle(), pos = code.getRight();
				int psid = permutation.get(sid), ppid = -1, ppos = -1;
				if (pid == -1) {
					ppos = pos;
				} else {
					ppid = permutation.get(pid);
					ppos = psid;
				}

				/**
				 * Find corresponding fid after permutation
				 */
				int index = -1;
				for (Triple<Integer, Integer, Integer> codes : codeBook) {
					index++;
					if (codes.getLeft() == psid && codes.getMiddle() == ppid && codes.getRight() == ppos)
						break;
				}
				if (index == -1)
					continue;

				transfer[i][k] = index;
			}
		}

		smile.classification.LogisticRegression.Trainer trainerLR = null;
		trainerLR = new smile.classification.LogisticRegression.Trainer();
		smile.classification.LogisticRegression algoLR = trainerLR.train(trainset.getLeft(), trainset.getRight());

		/**
		 * Learned Rule (wrapped with a learned function / model)
		 */
		Function<Profile<Integer>, List<Integer>> mechanism = profile -> {
			List<Float> feature = DataEngine.getFeatures(profile);
			double[] x = new double[feature.size()];
			for (int i = 0; i < feature.size(); i++)
				x[i] = feature.get(i);

			int[] pred = new int[2 * nPermutation];
			for (int p = 0; p < nPermutation; p++)
				pred[p] = pdt.get(p).predict(x);

			for (int i = 0; i < nPermutation; i++) {
				List<Integer> permutation = permutations.get(i);
				for (int k = 0; k < dim; k++)
					x[k] = feature.get(transfer[i][k]);
				pred[i + nPermutation] = permutation.get(algoLR.predict(x));
			}
			List<Integer> winners = new ArrayList<>();
			int[] modes = MathLib.Data.mode(pred);
			for (int mode : modes)
				winners.add(mode);
			return winners;
		};
		LearnedRule learnedRule = new LearnedRule(mechanism);

		StringBuffer sb = new StringBuffer();
		sb.append(meta);
		sb.append("Permuted Decision Tree and Permuted Logistic Regression - ");
		sb.append("DT (depth = " + depth + ")");
		learnedRule.setName(sb.toString());

		sb = new StringBuffer();
		for (int i = 0; i < pdt.size(); i++)
			sb.append(pdt.get(i).toString() + "\n");

		try {
			FileUtils.write(new File("csc/model.txt"), sb.toString(), "UTF8", true);
		} catch (IOException e) {
			e.printStackTrace();
		}
		return learnedRule;
	}

	/**
	 * create ensemble rule with some individual rules
	 * 
	 * @param rules
	 * @return ensemble rule
	 */
	public <T> VotingRule getEnsembleRule(List<VotingRule> rules) {
		// ensemble voting rule
		Function<Profile<T>, List<T>> mechanism = profile -> {
			List<T> choices = rules.stream().flatMap(r -> {
				List<T> winnerList = r.getAllWinners(profile);
				if (winnerList == null)
					return null;
				else
					return winnerList.stream();
			}).filter(Objects::nonNull).collect(Collectors.toList());
			return choices = MathLib.Data.mode(choices);
		};
		VotingRule ensembleRule = new LearnedRule(mechanism);
		return ensembleRule;
	}

	/**
	 * Permutated Decision Tree and Logistic Regression Rule
	 * 
	 * @param profiles
	 * @return
	 */
	public VotingRule getNeuralDecisionTreeRule(List<ChoiceTriple<Integer>> profiles, int depth) {
		Pair<double[][], int[]> trainset = DataEngine.getFlatDataSet(profiles);
		double[][] features = trainset.getLeft();

		NeuralDecisionTree ndt = new NeuralDecisionTree(depth);
		ndt.train(features, trainset.getRight());

		/**
		 * Learned Rule (wrapped with a learned function / model)
		 */
		Function<Profile<Integer>, List<Integer>> mechanism = profile -> {
			List<Float> feature = DataEngine.getFeatures(profile);
			double[] x = new double[feature.size()];
			for (int i = 0; i < feature.size(); i++)
				x[i] = feature.get(i);

			List<Integer> winners = new ArrayList<>();
			winners.add(ndt.predict(x));
			return winners;
		};
		LearnedRule learnedRule = new LearnedRule(mechanism);

		StringBuffer sb = new StringBuffer();
		sb.append(meta);
		sb.append("Neural Decision Tree - ");
		sb.append("DT (depth = " + depth + ")");
		learnedRule.setName(sb.toString());

		sb = new StringBuffer();
		try {
			FileUtils.write(new File("csc/model.txt"), sb.toString(), "UTF8", true);
		} catch (IOException e) {
			e.printStackTrace();
		}
		return learnedRule;
	}

	public VotingRule getNeuralDecisionForestRule(List<ChoiceTriple<Integer>> profiles, int depth) {
		Pair<double[][], int[]> trainset = DataEngine.getFlatDataSet(profiles);
		double[][] features = trainset.getLeft();

		int dim = features[0].length;

		int nOutput = (int) Arrays.stream(trainset.getValue()).distinct().count();
		NeuralDecisionForest.Trainer trainer = null;
		trainer = new NeuralDecisionForest.Trainer(ErrorFunction.CROSS_ENTROPY, dim, 10, nOutput);
		NeuralDecisionForest algo = trainer.train(features, trainset.getValue());

		/**
		 * Learned Rule (wrapped with a learned function / model)
		 */
		Function<Profile<Integer>, List<Integer>> mechanism = profile -> {
			List<Float> feature = DataEngine.getFeatures(profile);

			double[] x = new double[feature.size()];
			for (int i = 0; i < feature.size(); i++)
				x[i] = feature.get(i);

			List<Integer> winners = new ArrayList<>();
			winners.add(algo.predict(x));
			return winners;
		};
		LearnedRule learnedRule = new LearnedRule(mechanism);

		StringBuffer sb = new StringBuffer();
		sb.append(meta);
		sb.append("Neural Decision Forest - ");
		sb.append("DT (depth = " + depth + ")");
		learnedRule.setName(sb.toString());

		sb = new StringBuffer();
		try {
			FileUtils.write(new File("csc/model.txt"), sb.toString(), "UTF8", true);
		} catch (IOException e) {
			e.printStackTrace();
		}
		return learnedRule;
	}

	/**
	 * @param profiles
	 * @param depth
	 * @return voting rule learned using maximum cut plane
	 */
	public VotingRule getMaximumCutPlaneRule(List<ChoiceTriple<Integer>> profiles, int depth) {
		Pair<double[][], int[]> trainset = DataEngine.getFlatDataSet(profiles);
		double[][] input = trainset.getLeft();
		int[] y = trainset.getRight();

		MaximumCutPlane algo = new MaximumCutPlane();
		algo.setLearningRate(1.0E-6).setMaxDepth(depth).train(input, y);
		algo.setOptUpdateAlgo(OptUpdateAlgo.SGD).setBatchSize(100);

		/**
		 * Learned Rule (wrapped with a learned function / model)
		 */
		Function<Profile<Integer>, List<Integer>> mechanism = profile -> {
			List<Float> feature = DataEngine.getFeatures(profile);
			double[] x = new double[feature.size()];
			for (int i = 0; i < feature.size(); i++)
				x[i] = feature.get(i);

			List<Integer> winners = new ArrayList<>();
			winners.add(algo.predict(x));
			return winners;
		};
		LearnedRule learnedRule = new LearnedRule(mechanism);

		StringBuffer sb = new StringBuffer();
		sb.append("MaxCut Plane - DT (max_depth = ").append(depth).append(")");
		learnedRule.setName(sb.toString());

		sb.append("\n").append(algo.toString()).append("\n");
		try {
			FileUtils.write(new File(".csc/MaximumCutPlane.txt"), sb.toString(), "UTF8", true);
		} catch (IOException e) {
			e.printStackTrace();
		}
		return learnedRule;
	}

	public VotingRule getMaximumCutPlaneRule(List<ChoiceTriple<Integer>> profiles, VotingRule oracle, int depth) {
		Pair<double[][], int[]> trainset = DataEngine.getFlatDataSet(profiles);
		double[][] input = trainset.getLeft();
		int[] y = trainset.getRight();

		MaximumCutPlane algo = new MaximumCutPlane();
		algo.setLearningRate(1.0E-6).setMaxDepth(depth);
		algo.train(input, y);

		Function<Profile<Integer>, List<Integer>> mechanism = profile -> {
			List<Float> feature = DataEngine.getFeatures(profile);
			double[] x = new double[feature.size()];
			for (int i = 0; i < feature.size(); i++)
				x[i] = feature.get(i);

			List<Integer> winners = new ArrayList<>();
			winners.add(algo.predict(x));
			return winners;
		};
		LearnedRule learnedRule = new LearnedRule(mechanism);

		StringBuffer sb = new StringBuffer();
		sb.append("MaxCut Plane - DT (max_depth = ").append(depth).append(")");
		learnedRule.setName(sb.toString());

		sb.append("\n").append(algo.toString()).append("\n");
		try {
			FileUtils.write(new File("csc/MaximumCutPlane-" + oracle.toString() + ".txt"), sb.toString(), "UTF8", true);
		} catch (IOException e) {
			e.printStackTrace();
		}
		return learnedRule;
	}

	public VotingRule getLinearMachineRule(List<ChoiceTriple<Integer>> profiles, VotingRule oracle) {
		Pair<double[][], int[]> trainset = DataEngine.getFlatDataSet(profiles);
		double[][] input = trainset.getLeft();
		int[] y = trainset.getRight();

		LinearMachine algo = new LinearMachine();
		algo.setLearningRate(1.0E-3F).setMaxIter(3000).train(input, y);

		Function<Profile<Integer>, List<Integer>> mechanism = profile -> {
			List<Float> feature = DataEngine.getFeatures(profile);
			double[] x = new double[feature.size()];
			for (int i = 0; i < feature.size(); i++)
				x[i] = feature.get(i);

			List<Integer> winners = new ArrayList<>();
			winners.add(algo.predict(x));
			return winners;
		};
		LearnedRule learnedRule = new LearnedRule(mechanism);

		StringBuffer sb = new StringBuffer();
		sb.append("LinearMachine");
		learnedRule.setName(sb.toString());

		sb.append("\n").append(algo.toString()).append("\n");
		try {
			FileUtils.write(new File("csc/LinearMachine-" + oracle.toString() + ".txt"), sb.toString(), "UTF8", true);
		} catch (IOException e) {
			e.printStackTrace();
		}
		return learnedRule;
	}

	public double[] getMultiClassSVMRule(List<ChoiceTriple<Integer>> profiles, double c, int numItem, int[] numVotes,
			VotingRule oracle) throws InterruptedException, IOException {
		String base = "/Users/chjiang/GitHub/svm/svm_multiclass/";
		DataEngine.getSVMDataSet(profiles, base + oracle.toString() + "-train.dat");

		StringBuffer sb = new StringBuffer();
		sb.append(base).append("svm_multiclass_learn -c ").append(c).append(" ");
		sb.append(base).append(oracle.toString()).append("-train.dat").append(" ");
		sb.append(base).append(oracle.toString()).append("-model.dat");

		Process process = Runtime.getRuntime().exec(sb.toString());
		process.waitFor();

		if (process.exitValue() != 0) {
			System.err.println("ERROR: Training Process Failed to Exit.");
			return null;
		}

		StringBuffer summary = new StringBuffer();
		summary.append(oracle.toString());

		double[] accuracies = new double[numVotes.length];
		for (int v = 0; v < numVotes.length; v++) {
			int numVote = numVotes[v];

			List<Integer> truthList = new ArrayList<>();
			DataEngine.getSVMTestingSet(numItem, numVote, oracle, base + oracle.toString() + "-test.dat", truthList);
			sb = new StringBuffer();
			sb.append(base).append("svm_multiclass_classify").append(" ");
			sb.append(base).append(oracle.toString()).append("-test.dat").append(" ");
			sb.append(base).append(oracle.toString()).append("-model.dat").append(" ");
			sb.append(base).append(oracle.toString()).append("-predict.dat");

			process = Runtime.getRuntime().exec(sb.toString());
			process.waitFor();

			if (process.exitValue() != 0) {
				System.err.println("ERROR: Testing Process Failed to Exit.");
				accuracies[v] = -1;
				continue;
			}

			String src = base + oracle.toString() + "-predict.dat";
			List<double[]> data = Data.loadData(src, " ");
			int count = 0;
			for (int i = 0; i < data.size(); i++) {
				int truth = truthList.get(i);
				if (truth == data.get(i)[0])
					count++;
			}
			accuracies[v] = count * 1.0 / data.size();
			summary.append("\t" + accuracies[v]);
		}
		summary.append("\n");
		FileUtils.writeStringToFile(new File(base + oracle.toString() + "-perf.dat"), summary.toString(), "UTF8", true);
		return accuracies;
	}
}