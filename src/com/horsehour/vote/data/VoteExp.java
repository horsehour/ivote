package com.horsehour.vote.data;

import java.io.IOException;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.OpenOption;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiFunction;

import org.apache.commons.lang3.SerializationUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import com.horsehour.util.MathLib;
import com.horsehour.util.TickClock;
import com.horsehour.vote.Profile;
import com.horsehour.vote.train.VoteAdaBoost;
import com.horsehour.vote.train.VoteNetwork;

import smile.classification.LogisticRegression;
import smile.classification.NeuralNetwork;
import smile.classification.RandomForest;
import smile.classification.SoftClassifier;
import smile.math.Math;

/**
 * @author Chunheng Jiang
 * @since Mar 19, 2017
 * @version 1.0
 */
public class VoteExp {
	public static OpenOption[] options =
			{ StandardOpenOption.APPEND, StandardOpenOption.CREATE, StandardOpenOption.WRITE };

	public double cutoff = 0.3;

	/**
	 * Soft cutoff predictor works as follows: if the sum of the top
	 * alternatives gets certain percentage of the sum value, they will be
	 * elected as the winners. Here, the probability is the posterior
	 * probability.
	 */
	public BiFunction<List<Integer>, double[], List<Integer>> softCutoff = (items, probability) -> {
		List<Integer> pred = new ArrayList<>();

		double alpha = MathLib.Data.sum(probability) * cutoff;
		int[] rank = MathLib.getRank(probability, false);
		// System.out.print(Arrays.toString(rank) + "\t");

		double subsum = 0;
		for (int r : rank) {
			int ind = items.indexOf(r);
			if (ind == -1)
				continue;

			pred.add(r);
			subsum += probability[r];
			if (subsum >= alpha)
				break;
		}
		return pred;
	};

	public BiFunction<List<Integer>, double[], List<Integer>> softCutoff2 = (items, probability) -> {
		Map<Double, List<Integer>> tiers = new TreeMap<>();
		double alpha = 0, sum = 0;
		for (int i : items) {
			double pro = probability[i];
			List<Integer> member = tiers.get(pro);
			if (member == null)
				member = new ArrayList<>();
			member.add(i);
			tiers.put(pro, member);
			sum += pro;
		}
		alpha = sum * cutoff;

		List<Integer> pred = new ArrayList<>();
		List<Double> distinct = new ArrayList<>(tiers.keySet());
		double subsum = 0;
		for (int i = distinct.size() - 1; i >= 0; i--) {
			double score = distinct.get(i);
			List<Integer> value = tiers.get(score);
			subsum += score * value.size();
			if (subsum > alpha)
				break;
			pred.addAll(value);
		}
		Collections.sort(pred);
		return pred;
	};

	/**
	 * Hard cutoff predictor only select those alternatives with the maximum
	 * posterior probability
	 */
	public BiFunction<List<Integer>, double[], List<Integer>> hardCutoff = (items, probability) -> {
		List<Integer> pred = new ArrayList<>();
		double max = -100;
		for (int i : items) {
			if (probability[i] > max) {
				pred.clear();
				max = probability[i];
				pred.add(i);
			} else if (probability[i] == max) {
				pred.add(i);
			}
		}
		return pred;
	};

	/**
	 * Compute three performance indicators: Precision, Recall and F-score
	 * 
	 * @param truth
	 * @param pred
	 * @return three performance measures
	 */
	public float[] getPerformance(List<List<Integer>> truth, List<List<Integer>> pred) {
		int n = truth.size(), nWinner = 0, nElected = 0, nHit = 0;
		List<Integer> winners = null, elected = null;
		for (int i = 0; i < n; i++) {
			winners = truth.get(i);
			elected = pred.get(i);

			if (winners == null || winners.isEmpty() || elected == null || elected.isEmpty())
				continue;

			nWinner += winners.size();
			nElected += elected.size();

			for (int elect : elected) {
				if (winners.contains(elect))
					nHit++;
			}
		}

		float[] perf = new float[3];
		perf[0] = nHit * 1.0f / nElected;
		perf[1] = nHit * 1.0f / nWinner;
		perf[2] = 2 * perf[0] * perf[1] / (perf[0] + perf[1]);
		return perf;
	}

	/**
	 * Split data set into training set and evaluation set
	 * 
	 * @param data
	 * @param dataset
	 * @param ratioTrain
	 * @throws IOException
	 */
	void split(Path data, List<Pair<double[][], int[][]>> dataset, float ratioTrain) throws IOException {
		int nTotal = (int) Files.lines(data).count();
		int nTrain = (int) (nTotal * ratioTrain);
		int nEval = nTotal - nTrain;

		List<Integer> trainList = MathLib.Rand.sample(0, nTotal, nTrain);
		Collections.sort(trainList);

		double[][] trainset = new double[nTrain][];
		int[][] train_label = new int[nTrain][];
		double[][] evalset = new double[nEval][];
		int[][] eval_label = new int[nEval][];

		AtomicInteger ind = new AtomicInteger(0);
		AtomicInteger cTrain = new AtomicInteger(0);
		AtomicInteger cEval = new AtomicInteger(0);

		Files.lines(data).forEach(line -> {
			String[] fields = line.split("\t|;");
			String[] columns = fields[0].replaceAll("\\[|\\]| ", "").split(",");
			int dim = columns.length;
			double[] inputs = new double[dim];
			for (int k = 0; k < dim; k++)
				inputs[k] = Double.parseDouble(columns[k]);

			columns = fields[1].replaceAll("\\[|\\]| ", "").split(",");

			int[] outputs = FeatureLab.decode.apply(columns);

			if (!trainList.isEmpty() && ind.get() == trainList.get(0)) {
				trainset[cTrain.get()] = inputs;
				train_label[cTrain.get()] = outputs;
				trainList.remove(0);
				cTrain.getAndIncrement();
			} else {
				evalset[cEval.get()] = inputs;
				eval_label[cEval.get()] = outputs;
				cEval.getAndIncrement();
			}
			ind.getAndIncrement();
		});
		dataset.add(Pair.of(trainset, train_label));
		dataset.add(Pair.of(evalset, eval_label));
	}

	/**
	 * Flatten samples with multiple outputs to samples each of which has an
	 * unique output
	 * 
	 * @param dataset
	 * @return Flatten DataSet
	 */
	Pair<double[][], int[]> flatten(Pair<double[][], int[][]> dataset) {
		int nTotal = 0;

		for (int[] label : dataset.getRight())
			nTotal += label.length;

		double[][] inputs = new double[nTotal][];
		int[] outputs = new int[nTotal];

		double[][] in = dataset.getKey();
		int[][] out = dataset.getValue();

		int c = 0;
		for (int i = 0; i < in.length; i++) {
			for (int o : out[i]) {
				outputs[c] = o;
				inputs[c] = in[i];
				c++;
			}
		}
		return Pair.of(inputs, outputs);
	}

	/**
	 * @param trainset
	 * @param items
	 * @param nHidden
	 * @return Neural Network
	 */
	public NeuralNetwork getNeuralNetwork(Pair<double[][], int[]> trainset, List<Integer> items, int nHidden) {
		double[][] inputs = trainset.getLeft();
		int[] output = trainset.getRight();
		int dim = inputs[0].length;
		int nInput = dim, nOutput = items.size();

		NeuralNetwork.ErrorFunction errorFunc = NeuralNetwork.ErrorFunction.LEAST_MEAN_SQUARES;
		NeuralNetwork.Trainer trainer = new NeuralNetwork.Trainer(errorFunc, nInput, nHidden, nOutput);
		NeuralNetwork algo = trainer.train(inputs, output);
		return algo;
	}

	/**
	 * @param trainset
	 * @param items
	 * @return Random Forest
	 */
	public RandomForest getRandomForest(Pair<double[][], int[]> trainset, List<Integer> items) {
		double[][] inputs = trainset.getLeft();
		int[] outputs = trainset.getRight();

		RandomForest.Trainer trainer = new RandomForest.Trainer();
		int depth = 1, nTree = 50;
		trainer.setNumTrees(nTree);
		trainer.setMaxNodes(2 * depth + 1);
		int dim = inputs[0].length;
		int nFeature = (int) Math.floor(Math.sqrt(dim) / 3);
		trainer.setNumRandomFeatures(nFeature);
		RandomForest algo = trainer.train(inputs, outputs);
		return algo;
	}

	public LogisticRegression getLogistic(Pair<double[][], int[]> trainset, List<Integer> items) {
		double[][] inputs = trainset.getLeft();
		int[] output = trainset.getRight();

		LogisticRegression.Trainer trainer = new LogisticRegression.Trainer();
		trainer.setRegularizationFactor(3.0);
		LogisticRegression algo = trainer.train(inputs, output);

		SerializationUtils.serialize((Serializable) algo);
		return algo;
	}

	public VoteAdaBoost getAdaBoost(Pair<double[][], int[]> trainset, List<Integer> items) {
		double[][] inputs = trainset.getLeft();
		int[] output = trainset.getRight();

		VoteAdaBoost.Trainer trainer = new VoteAdaBoost.Trainer();
		trainer.setNumTrees(50);
		trainer.setMaxNodes(2 * 3 + 1);
		VoteAdaBoost algo = trainer.train(inputs, output);
		return algo;
	}

	/**
	 * Learned voting rule based on neural network
	 * 
	 * @param trainset
	 * @param items
	 * @param nHidden
	 * @return learned rule
	 */
	public VoteNetwork getVoteNet(Pair<double[][], int[][]> trainset, List<Integer> items, int nHidden) {
		double[][] inputs = trainset.getLeft();
		int[][] output = trainset.getRight();
		int nInput = inputs[0].length, nOutput = items.size();

		VoteNetwork.ErrorFunction errorFunc = VoteNetwork.ErrorFunction.LEAST_MEAN_SQUARES;
		VoteNetwork.Trainer trainer = new VoteNetwork.Trainer(errorFunc, nInput, nHidden, nOutput);
		VoteNetwork algo = trainer.train(inputs, output);

		List<List<Integer>> truth = new ArrayList<>();
		List<List<Integer>> preds = new ArrayList<>();
		for (int i = 0; i < output.length; i++) {
			List<Integer> y = new ArrayList<>();
			for (int k = 0; k < output[i].length; k++)
				y.add(output[i][k]);
			truth.add(y);

			double[] input = inputs[i];
			double[] probability = new double[nOutput];
			algo.predict(input, probability);

			List<Integer> pred = softCutoff.apply(items, probability);
			if (pred.isEmpty())
				pred = hardCutoff.apply(items, probability);
			preds.add(pred);
		}
		return algo;
	}

	public MultiLayerNetwork getDNN(Pair<double[][], int[][]> trainset, List<Integer> items) {
		double[][] features = trainset.getLeft();
		int[][] winners = trainset.getRight();

		int dim = features[0].length, len = features.length, nItem = items.size();

		double[] winner = null;
		List<DataSet> dataset = new ArrayList<>();
		for (int i = 0; i < len; i++) {
			INDArray input = Nd4j.create(features[i]);

			winner = new double[items.size()];
			for (int w : winners[i])
				winner[w] = 1;

			INDArray output = Nd4j.create(winner);
			dataset.add(new DataSet(input, output));
		}

		int batch = dataset.size() / 20;
		DataSetIterator iter = new ListDataSetIterator(dataset, batch);

		int seed = 123;
		double learningRate = 0.001, momentum = 0.98;

		int nInput = dim, nOutput = nItem, nHidden = nItem, nRun = 10;

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(Updater.NESTEROVS)
				.iterations(1).activation("relu").weightInit(WeightInit.XAVIER).learningRate(learningRate)
				.momentum(momentum).regularization(true).l1(learningRate * 0.001).list()
				.layer(0, new DenseLayer.Builder().nIn(nInput).nOut(nHidden).build())
				.layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).activation("softmax").nIn(nHidden)
						.nOut(nOutput).build())
				.pretrain(false).backprop(true).build();

		MultiLayerNetwork algo = new MultiLayerNetwork(conf);
		algo.init();

		algo.setListeners(new ScoreIterationListener(5));
		for (int i = 0; i < nRun; i++)
			algo.fit(iter);

		for (int i = 0; i < len; i++)
			System.out.println(algo.output(Nd4j.create(features[i])));
		return algo;
	}

	public SoftClassifier<double[]> algo = null;
	public int m = 0;

	public BiFunction<double[], List<Integer>, List<Integer>> mechanism = (x, items) -> {
		double[] probability = new double[m];
		algo.predict(x, probability);
		List<Integer> pred = softCutoff.apply(items, probability);
		if (pred.isEmpty())
			pred = hardCutoff.apply(items, probability);
		return pred;
	};

	public static void main(String[] args) throws IOException {
		TickClock.beginTick();

		VoteExp exp = new VoteExp();
		int m = 30;
		exp.m = m;

		String base = "/users/chjiang/github/csc/";
		Path data = Paths.get(base + "/M10-30.csv");

		List<Pair<double[][], int[][]>> dataset = new ArrayList<>();

		String name = "logistic";
		Path model = Paths.get(base + name + ".m." + m + ".mdl");
		if (!Files.exists(model)) {
			exp.split(data, dataset, 0.7F);

			Pair<double[][], int[]> samples = exp.flatten(dataset.get(0));
			List<Integer> items = new ArrayList<>();
			for (int i = 0; i < m; i++)
				items.add(i);

			if (name.contains("neuralnetwork"))
				exp.algo = exp.getNeuralNetwork(samples, items, m);
			else if (name.contains("votenet"))
				exp.algo = exp.getVoteNet(dataset.get(0), items, m);
			else if (name.contains("randomforest"))
				exp.algo = exp.getRandomForest(samples, items);
			else if (name.contains("logistic")) {
				exp.cutoff = 0.5;
				exp.algo = exp.getLogistic(samples, items);
			} else if (name.contains("adaboost")) {
				exp.cutoff = 0.3;
				exp.algo = exp.getAdaBoost(samples, items);
			}

			Files.write(model, SerializationUtils.serialize(exp.algo), options);

			double[][] inputs = dataset.get(1).getKey();
			int[][] outputs = dataset.get(1).getValue();

			List<List<Integer>> truth = new ArrayList<>();
			List<List<Integer>> pred = new ArrayList<>();

			float upperRecall = 0.0F, upperF1 = 0.0F;
			for (int i = 0; i < inputs.length; i++) {
				double[] input = inputs[i];
				List<Integer> winners = new ArrayList<>();
				for (int k = 0; k < outputs[i].length; k++)
					winners.add(outputs[i][k]);

				truth.add(winners);
				float recall = 1.0F / winners.size();
				upperRecall += recall;
				upperF1 += 2 * recall / (1 + recall);
				System.out.print(winners + "\t");
				List<Integer> predict = exp.mechanism.apply(input, items);
				System.out.println(predict);
				pred.add(predict);
			}
			upperRecall /= inputs.length;
			upperF1 /= inputs.length;
			float[] perf = exp.getPerformance(truth, pred);
			System.out.println(Arrays.toString(perf) + "\t" + upperRecall + "\t" + upperF1);
		}

//		exp.algo = SerializationUtils.deserialize(Files.readAllBytes(model));
//		List<String> lines = Files.readAllLines(Paths.get(base + "winners-stv-soc3.txt"));
//		Path file = null;
//		List<Integer> inclusive = null;
//		Profile<Integer> profile = null;
//
//		List<List<Integer>> winnerList = new ArrayList<>();
//		List<List<Integer>> predList = new ArrayList<>();
//
//		StringBuffer sb = new StringBuffer();
//		for (String line : lines) {
//			int ind = line.indexOf("\t");
//			String fnm = line.substring(0, ind);
//			int numItem = Integer.parseInt(fnm.substring(1, fnm.indexOf("N")));
//			if (numItem > m)
//				continue;
//			else {
//				file = Paths.get(base + "soc-3-stv/" + fnm);
//				if (!Files.exists(file))
//					file = Paths.get(base + "soc-3-hardcase/" + fnm);
//			}
//			profile = DataEngine.loadProfile(file);
//
//			inclusive = new ArrayList<>();
//			for (int i = 0; i < numItem; i++)
//				inclusive.add(i);
//
//			double[] features = FeatureLab.getF1(profile, inclusive, m);
//			List<Integer> pred = exp.mechanism.apply(features, inclusive);
//			String binary = line.substring(ind + 1).trim();
//			List<Integer> winners = FeatureLab.hotDecode.apply(binary);
//
//			sb.append(fnm + "\t").append(winners).append("\t").append(pred).append("\n");
//
//			winnerList.add(winners);
//			predList.add(pred);
//		}
//		sb.append(Arrays.toString(exp.getPerformance(winnerList, predList))).append("\n");
//		Files.write(Paths.get(base + name + "-soc-3.pred"), sb.toString().getBytes(), options);

		TickClock.stopTick();
	}

	public void getDNNPred() throws IOException {
		TickClock.beginTick();

		VoteExp exp = new VoteExp();
		int m = 30;
		exp.m = m;

		String base = "/users/chjiang/github/csc/";
		Path data = Paths.get(base + "/M10-30.csv");

		List<Pair<double[][], int[][]>> dataset = new ArrayList<>();

		exp.split(data, dataset, 0.7F);

		List<Integer> items = new ArrayList<>();
		for (int i = 0; i < m; i++)
			items.add(i);

		MultiLayerNetwork dnn = exp.getDNN(dataset.get(0), items);

		List<String> lines = Files.readAllLines(Paths.get(base + "winners-stv-soc3.txt"));

		Path file = null;
		List<Integer> inclusive = null;
		Profile<Integer> profile = null;

		List<List<Integer>> winnerList = new ArrayList<>();
		List<List<Integer>> predList = new ArrayList<>();

		StringBuffer sb = new StringBuffer();
		for (String line : lines) {
			int ind = line.indexOf("\t");
			String fnm = line.substring(0, ind);
			int numItem = Integer.parseInt(fnm.substring(1, fnm.indexOf("N")));
			if (numItem > m)
				continue;
			else {
				file = Paths.get(base + "soc-3-stv/" + fnm);
				if (!Files.exists(file))
					file = Paths.get(base + "soc-3-hardcase/" + fnm);
			}
			profile = DataEngine.loadProfile(file);

			inclusive = new ArrayList<>();
			for (int i = 0; i < numItem; i++)
				inclusive.add(i);

			double[] features = FeatureLab.getF1(profile, inclusive, m);

			int[] output = dnn.predict(Nd4j.create(features));
			List<Integer> pred = new ArrayList<>();
			for (int i = 0; i < output.length; i++)
				pred.add(output[i]);

			String binary = line.substring(ind + 1).trim();
			List<Integer> winners = FeatureLab.hotDecode.apply(binary);

			sb.append(fnm + "\t").append(winners).append("\t").append(pred).append("\n");

			winnerList.add(winners);
			predList.add(pred);
		}

		sb.append(Arrays.toString(exp.getPerformance(winnerList, predList))).append("\n");
		Files.write(Paths.get(base + "dnn-soc-3.pred"), sb.toString().getBytes(), options);

		TickClock.stopTick();
	}

	public static void main2(String[] args) throws IOException {
		TickClock.beginTick();

		VoteExp exp = new VoteExp();
		exp.getDNNPred();

		TickClock.stopTick();
	}
}