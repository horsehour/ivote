package com.horsehour.vote.train;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.BiFunction;

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

import com.horsehour.vote.Profile;
import com.horsehour.vote.rule.Condorcet;
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
	 * @param k
	 *            number of profiles
	 * @return Training profiles
	 */
	public List<Profile<Integer>> generateTrainProfiles(int m, int n, int k) {
		
		return null;
	}

	/**
	 * Extract input features from a given profile
	 * 
	 * @param profile
	 * @param items
	 * @return head-to-head competition outcome
	 */
	double[] getFeatures(Profile<Integer> profile, List<Integer> items) {
		int m = items.size();
		int dim = (m * (m - 1)) / 2;
		double[] features = new double[dim];
		int[][] pmm = Condorcet.getPairwisePreferenceMatrix(profile, items.toArray(new Integer[0]));
		int d = 0;
		for (int i = 0; i < m; i++) {
			for (int j = i + 1; j < m; j++) {
				features[d] = pmm[i][j];
				d++;
			}
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
	public VotingRule getDeepLearningRule(List<Profile<Integer>> profiles, List<Integer> items, VotingRule oracle) {
		Pair<List<double[]>, List<int[]>> trainset = getTrainInstances(profiles, items, oracle);

		List<double[]> features = trainset.getLeft();
		List<int[]> winners = trainset.getRight();
		int dim = features.get(0).length, len = features.size(), nItem = items.size();

		List<DataSet> dataset = new ArrayList<>();
		for (int i = 0; i < len; i++) {
			INDArray input = Nd4j.create(features.get(i));
			INDArray output = Nd4j.zeros(1, nItem);
			output.putScalar(winners.get(i), 1);
			// output.putScalar(0, trainset.getRight()[i][0], 1);
			dataset.add(new DataSet(input, output));
		}

		int batch = dataset.size() / 20;
		DataSetIterator iter = new ListDataSetIterator(dataset, batch);

		int seed = 123;
		double learningRate = 0.001, momentum = 0.98;

		int nInput = dim, nOutput = nItem, nHidden = 20, nRun = 100;

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(Updater.NESTEROVS).iterations(1).activation("relu").weightInit(WeightInit.XAVIER).learningRate(learningRate).momentum(momentum).regularization(true).l1(learningRate * 0.005).list().layer(0, new DenseLayer.Builder().nIn(nInput).nOut(nHidden).build()).layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).activation("softmax").nIn(nHidden).nOut(nOutput).build()).pretrain(false).backprop(true).build();

		MultiLayerNetwork algo = new MultiLayerNetwork(conf);
		algo.init();
		algo.setListeners(new ScoreIterationListener(5));

		for (int i = 0; i < nRun; i++) {
			algo.fit(iter);
		}

		BiFunction<Profile<Integer>, List<Integer>, List<Integer>> agent = (profile, cands) -> {
			double[] x = getFeatures(profile, cands);
			int[] ret = algo.predict(Nd4j.create(x));
			return Arrays.asList(ret[0]);
		};

		LearnedRule learnedRule = new LearnedRule(agent);
		StringBuffer sb = new StringBuffer();
		sb.append("DL (").append("nhidden = " + nHidden).append(", lr = " + learningRate).append(", momentum = " + momentum).append(", optimizer = sgd").append(", activation = relu|softmax").append(", l1 = true").append(", lossfunction = negativeloglikelihood)");
		learnedRule.setName(sb.toString());

		try {
			FileUtils.write(new File("csc/model.txt"), algo.conf().toJson() + "\n", "UTF8", true);
		} catch (IOException e) {
			e.printStackTrace();
		}

		return learnedRule;
	}

	public static void main(String[] args) {

	}
}
