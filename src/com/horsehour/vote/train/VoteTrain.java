package com.horsehour.vote.train;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.OpenOption;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.BiPredicate;

import org.apache.commons.io.FileUtils;
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

import com.horsehour.util.TickClock;
import com.horsehour.vote.DataEngine;
import com.horsehour.vote.Profile;
import com.horsehour.vote.rule.Borda;
import com.horsehour.vote.rule.LearnedRule;
import com.horsehour.vote.rule.VotingRule;

/**
 * @author Chunheng Jiang
 * @since Mar 15, 2017
 * @version 1.0
 */
public class VoteTrain {
	/**
	 * Generate training profiles with specific parameters
	 * 
	 * @param m
	 *            number of alternatives
	 * @param n
	 *            number of voters
	 * @param s
	 *            number of profiles
	 * @return Training profiles
	 */
	public List<Profile<Integer>> generateTrainProfiles(int m, int n, int s) {
		int numAEC = DataEngine.getAECStat(m, n).size();
		boolean compact = false;
		double ratio = s * 1.0 / numAEC;
		int nSample = (int) (numAEC * ratio);
		List<Profile<Integer>> profiles = new ArrayList<>();
		for (int i = 0; i < nSample; i++)
			profiles.add(DataEngine.getRandomProfile(m, n, compact));
		return profiles;
	}

	/**
	 * Extract input features from a given profile
	 * 
	 * @param profile
	 * @param items
	 * @return head-to-head competition outcome
	 */
	double[] getFeatures(Profile<Integer> profile, List<Integer> items) {
		int m = items.size(), n = profile.numVoteTotal;
		int dim = n * (m * (m - 1)) / 2, c = 0;
		double[] features = new double[dim];
		for (Integer[] preference : profile.data) {
			List<Integer> indices = new ArrayList<>();
			for (int pos : preference) {
				int ind = items.indexOf(pos);
				if (ind == -1)
					continue;
				indices.add(ind);
			}

			int[][] matrix = new int[m][m];
			for (int i = 0; i < m; i++) {
				int a = indices.get(i);
				for (int j = i + 1; j < m; j++) {
					int b = indices.get(j);
					matrix[a][b] = 1;
				}
			}

			for (int i = 0; i < m; i++)
				for (int j = i + 1; j < m; j++)
					features[c++] = matrix[i][j];
		}
		return features;
	}

	/**
	 * Extract input features from a set of profiles
	 * 
	 * @param profiles
	 * @param items
	 * @return input features of a set of profiles
	 */
	List<double[]> getInputFeatures(List<Profile<Integer>> profiles, List<Integer> items) {
		List<double[]> inputs = new ArrayList<>();
		for (Profile<Integer> profile : profiles)
			inputs.add(getFeatures(profile, items));
		return inputs;
	}

	/**
	 * Construct training instances from original profiles
	 * 
	 * @param profiles
	 * @param items
	 * @param oracle
	 * @return training instances with multiple output
	 */
	Pair<List<double[]>, List<int[]>> getTrainInstances(List<Profile<Integer>> profiles, List<Integer> items,
			VotingRule oracle) {
		List<double[]> inputs = new ArrayList<>();
		List<int[]> outputs = new ArrayList<>();
		for (Profile<Integer> profile : profiles) {
			List<Integer> winners = oracle.getAllWinners(profile);
			if (winners == null || winners.size() == 0)
				continue;

			inputs.add(getFeatures(profile, items));
			int nw = winners.size();
			int[] output = new int[nw];
			for (int i = 0; i < nw; i++)
				output[i] = winners.get(i);
			outputs.add(output);
		}
		return Pair.of(inputs, outputs);
	}

	/**
	 * Learning voting rule based on deep network
	 * 
	 * @param profiles
	 * @param items
	 * @param oracle
	 *            voting rule from which the algorithm is trying to learn
	 * @return learned voting rule
	 */
	public LearnedRule getDeepLearningRule(List<Profile<Integer>> profiles, List<Integer> items, VotingRule oracle) {
		Pair<List<double[]>, List<int[]>> trainset = getTrainInstances(profiles, items, oracle);

		List<double[]> features = trainset.getLeft();
		List<int[]> winners = trainset.getRight();
		int dim = features.get(0).length, len = features.size(), nItem = items.size();

		List<DataSet> dataset = new ArrayList<>();
		for (int i = 0; i < len; i++) {
			INDArray input = Nd4j.create(features.get(i));
			INDArray output = Nd4j.zeros(1, nItem);
			for (int w : winners.get(i))
				output.putScalar(0, w, 1);
			dataset.add(new DataSet(input, output));
		}

		int batch = dataset.size() / 10;
		DataSetIterator iter = new ListDataSetIterator(dataset, batch);

		int seed = 123;
		double learningRate = 0.001, momentum = 0.9;

		int nInput = dim, nOutput = nItem, nHidden = 10, nRun = 1;

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(Updater.NESTEROVS)
				.iterations(1).activation("relu").weightInit(WeightInit.XAVIER).learningRate(learningRate)
				.momentum(momentum).regularization(true).l1(2).list()
				.layer(0, new DenseLayer.Builder().nIn(nInput).nOut(nHidden).build())
				.layer(1, new DenseLayer.Builder().nIn(nHidden).nOut(nHidden).build())
				.layer(2, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).activation("softmax").nIn(nHidden)
						.nOut(nOutput).build())
				.pretrain(false).backprop(true).build();

		MultiLayerNetwork algo = new MultiLayerNetwork(conf);
		algo.init();
		algo.setListeners(new ScoreIterationListener(5));

		for (int i = 0; i < nRun; i++) {
			algo.fit(iter);
		}

		BiFunction<Profile<Integer>, List<Integer>, List<Integer>> agent = (profile, cands) -> {
			double[] x = getFeatures(profile, cands);
			int[] ret = algo.predict(Nd4j.create(x));
			List<Integer> pred = new ArrayList<>();
			for (int i : ret)
				pred.add(i);
			return pred;
		};

		LearnedRule learned = new LearnedRule(agent);
		StringBuffer sb = new StringBuffer();
		sb.append("DL (").append("nhidden = " + nHidden).append(", lr = " + learningRate)
				.append(", momentum = " + momentum).append(", optimizer = sgd").append(", activation = relu|softmax")
				.append(", l1 = true").append(", lossfunction = negativeloglikelihood)");
		learned.setName(sb.toString());

		try {
			FileUtils.write(new File("outcome/model.txt"), algo.conf().toJson() + "\n", "UTF8", true);
		} catch (IOException e) {
			e.printStackTrace();
		}
		return learned;
	}

	/**
	 * Single winner evaluation or comparison
	 */
	BiPredicate<List<Integer>, List<Integer>> match = (winners, predicted) -> {
		if (predicted == null)
			return false;
		return predicted.contains(winners.get(0));
	};

	public double getSimilarity(int m, int n, int s, VotingRule oracle, LearnedRule learned) {
		List<Profile<Integer>> profiles = generateTrainProfiles(m, n, s);
		List<Integer> items = new ArrayList<>();
		for (int i = 0; i < m; i++)
			items.add(i);

		int numTotal = 0, numMatch = 0;
		for (Profile<Integer> profile : profiles) {
			List<Integer> winners = oracle.getAllWinners(profile);
			if (winners == null || winners.size() == 0)
				continue;

			numTotal++;
			List<Integer> predicted = learned.getAllWinners(profile, items);
			if (match.test(winners, predicted))
				numMatch++;
		}
		return numMatch * 1.0d / numTotal;
	}

	public double eval(int m, int n, int s, VotingRule oracle, LearnedRule learned, Path report) {
		List<Profile<Integer>> profiles = generateTrainProfiles(m, n, s);
		List<Integer> items = new ArrayList<>();
		for (int i = 0; i < m; i++)
			items.add(i);

		StringBuffer sb = new StringBuffer();
		sb.append("truth\tpredict\n");

		int correct = 0, total = 0;
		for (Profile<Integer> profile : profiles) {
			List<Integer> winners = oracle.getAllWinners(profile);
			if (winners == null || winners.size() == 0)
				continue;

			total++;
			List<Integer> predicted = learned.getAllWinners(profile, items);
			if (winners.get(0) == predicted.get(0))
				correct++;

			sb.append(winners).append("\t").append(predicted).append("\n");
		}

		double accuracy = correct * 1.0d / total;
		sb.append("correct: ").append(accuracy).append("\n");

		OpenOption[] options =
				{ StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING };

		try {
			Files.write(report, sb.toString().getBytes(), options);
		} catch (IOException e) {
			e.printStackTrace();
		}
		return accuracy;
	}

	public static void main(String[] args) {
		TickClock.beginTick();

		int m = 3, s_train = 1000, s_eval = 1000;
		int[] votes = { 7, 9, 11 };

		List<Integer> items = new ArrayList<>();
		for (int i = 0; i < m; i++)
			items.add(i);

		VoteTrain vt = new VoteTrain();
		List<Profile<Integer>> profiles = null;

		VotingRule oracle = new Borda();
		LearnedRule learned = null;

		StringBuffer sb = new StringBuffer();
		
		for (int n : votes) {
			sb.append("\n");
			profiles = vt.generateTrainProfiles(m, n, s_train);
			learned = vt.getDeepLearningRule(profiles, items, oracle);
			String report = "outcome/M" + m + "N" + n + "S" + s_eval + "-Network.csv";
			sb.append(n + " : ").append(vt.eval(m, n, s_eval, oracle, learned, Paths.get(report)));
		}
		System.out.println(sb.toString());

		TickClock.stopTick();
	}
}
