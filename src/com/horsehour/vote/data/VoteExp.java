package com.horsehour.vote.data;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.OpenOption;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.BiFunction;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.tuple.Pair;

import com.horsehour.ml.classifier.tree.mtree.MaximumCutPlane;
import com.horsehour.util.MathLib;
import com.horsehour.util.TickClock;
import com.horsehour.vote.DataEngine;
import com.horsehour.vote.Profile;
import com.horsehour.vote.rule.Borda;
import com.horsehour.vote.rule.LearnedRule;
import com.horsehour.vote.rule.VotingRule;

import smile.classification.NeuralNetwork;

/**
 * @author Chunheng Jiang
 * @since Mar 15, 2017
 * @version 1.0
 */
public class VoteExp {
	public int fid = 2;

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
	public List<Profile<Integer>> generateProfiles(int m, int n, int s) {
		boolean compact = false;
		List<Profile<Integer>> profiles = new ArrayList<>();
		for (int i = 0; i < s; i++)
			profiles.add(DataEngine.getRandomProfile(m, n, compact));
		return profiles;
	}

	double[] getFeatures(Profile<Integer> profile, List<Integer> items) {
		if (fid == 1)
			return getFeatures1(profile, items);
		else
			return getFeatures2(profile, items);
	}

	/**
	 * Extract input features from a given profile
	 * 
	 * @param profile
	 * @param items
	 * @return head-to-head competition outcome
	 */
	double[] getFeatures1(Profile<Integer> profile, List<Integer> items) {
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

	double[] getFeatures2(Profile<Integer> profile, List<Integer> items) {
		int numItem = items.size();
		int numPreferences = profile.data.length;
		int[][] positions = new int[numItem][numPreferences];

		for (int i = 0; i < numItem; i++) {
			for (int k = 0; k < numPreferences; k++) {
				Integer[] preferences = profile.data[k];
				positions[i][k] = ArrayUtils.indexOf(preferences, items.get(i));
			}
		}

		int nv = profile.numVoteTotal;
		List<Float> features = new ArrayList<>();
		for (int i = 0; i < numItem; i++) {
			for (int j = i + 1; j < numItem; j++) {
				float[][] pairFeature = new float[numItem][numItem];
				for (int k = 0; k < numPreferences; k++) {
					int posI = positions[i][k];
					int posJ = positions[j][k];

					pairFeature[posI][posJ] += (profile.votes[k] * 1.0F / nv);
				}
				for (int idxI = 0; idxI < numItem; idxI++)
					for (int idxJ = 0; idxJ < numItem; idxJ++) {
						if (idxI == idxJ)
							continue;
						features.add(pairFeature[idxI][idxJ]);
					}
			}
		}

		double[] results = new double[features.size()];
		for (int i = 0; i < features.size(); i++)
			results[i] = features.get(i);
		return results;
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
	Pair<double[][], int[]> getTrainInstances(List<Profile<Integer>> profiles, List<Integer> items, VotingRule oracle) {
		List<double[]> features = new ArrayList<>();
		List<Integer> elected = new ArrayList<>();
		for (Profile<Integer> profile : profiles) {
			List<Integer> winners = oracle.getAllWinners(profile);
			if (winners == null || winners.size() == 0)
				continue;

			for (int winner : winners) {
				features.add(getFeatures(profile, items));
				elected.add(winner);
			}
		}

		int dim = features.get(0).length, len = features.size();
		double[][] inputs = new double[len][dim];
		int[] outputs = new int[len];
		for (int i = 0; i < len; i++) {
			inputs[i] = features.get(i);
			outputs[i] = elected.get(i);
		}
		return Pair.of(inputs, outputs);
	}

	/**
	 * Learned voting rule based on neural network
	 * 
	 * @param profiles
	 * @param items
	 * @param oracle
	 * @return learned voting rule
	 */
	public LearnedRule getNNRule(List<Profile<Integer>> profiles, List<Integer> items, VotingRule oracle) {
		Pair<double[][], int[]> trainset = getTrainInstances(profiles, items, oracle);

		double[][] inputs = trainset.getLeft();
		int[] output = trainset.getRight();
		int dim = inputs[0].length, nItem = items.size();
		int nInput = dim, nOutput = nItem, nHidden = nItem;

		NeuralNetwork.ErrorFunction errorFunc = NeuralNetwork.ErrorFunction.CROSS_ENTROPY;
		NeuralNetwork.Trainer trainer = null;
		trainer = new NeuralNetwork.Trainer(errorFunc, nInput, nHidden, nOutput);

		NeuralNetwork algo = trainer.train(inputs, output);

		List<List<Integer>> truth = new ArrayList<>();
		List<List<Integer>> preds = new ArrayList<>();
		for (int i = 0; i < output.length; i++) {
			truth.add(Arrays.asList(output[i]));
			double[] input = inputs[i];
			double[] pr = new double[nOutput];
			algo.predict(input, pr);
			int[] max = MathLib.argmax(pr);
			List<Integer> pred = new ArrayList<>();
			for (int item : max)
				pred.add(items.get(item));
			preds.add(pred);
		}
		System.out.println("Train Perf : " + Arrays.toString(getPerformance(truth, preds)));

		BiFunction<Profile<Integer>, List<Integer>, List<Integer>> agent = (profile, cands) -> {
			double[] x = getFeatures(profile, cands);
			double[] pr = new double[nOutput];
			algo.predict(x, pr);
			int[] max = MathLib.argmax(pr);
			List<Integer> pred = new ArrayList<>();
			for (int i : max)
				pred.add(cands.get(i));
			return pred;
		};
		LearnedRule learned = new LearnedRule(agent);
		return learned;
	}

	/**
	 * Learned voting rule based on maximum cut plane (MCP)
	 * 
	 * @param profiles
	 * @param items
	 * @param oracle
	 * @return learned voting rule
	 */
	public LearnedRule getMCPRule(List<Profile<Integer>> profiles, List<Integer> items, VotingRule oracle) {
		Pair<double[][], int[]> trainset = getTrainInstances(profiles, items, oracle);
		double[][] inputs = trainset.getLeft();
		int[] output = trainset.getRight();
		int depth = 5;
		MaximumCutPlane algo = new MaximumCutPlane();
		algo.setLearningRate(1.0E-6).setMaxDepth(depth);
		algo.train(inputs, output);

		List<List<Integer>> truth = new ArrayList<>();
		List<List<Integer>> preds = new ArrayList<>();
		for (int i = 0; i < output.length; i++) {
			truth.add(Arrays.asList(output[i]));
			double[] input = inputs[i];
			List<Integer> pred = new ArrayList<>();
			pred.add(algo.predict(input));
			preds.add(pred);
		}
		System.out.println("Train Perf : " + Arrays.toString(getPerformance(truth, preds)));

		BiFunction<Profile<Integer>, List<Integer>, List<Integer>> agent = (profile, cands) -> {
			double[] x = getFeatures(profile, cands);
			List<Integer> pred = new ArrayList<>();
			pred.add(algo.predict(x));
			return pred;
		};
		LearnedRule learned = new LearnedRule(agent);
		return learned;
	}

	/**
	 * Compute the performance of the learned rule
	 * 
	 * @param profiles
	 * @param oracle
	 * @param learned
	 * @return performance of the learned rule
	 */
	public float[] getPerformance(List<Profile<Integer>> profiles, VotingRule oracle, LearnedRule learned) {
		List<List<Integer>> truth = new ArrayList<>();
		List<List<Integer>> pred = new ArrayList<>();
		int m = profiles.get(0).getNumItem();
		List<Integer> items = new ArrayList<>();
		for (int i = 0; i < m; i++)
			items.add(i);

		for (Profile<Integer> profile : profiles) {
			truth.add(oracle.getAllWinners(profile));
			pred.add(learned.getAllWinners(profile, items));
		}
		return getPerformance(truth, pred);
	}

	/**
	 * Compute three performance indicators: Precision, Recall and F-score
	 * 
	 * @param truth
	 * @param pred
	 * @return three performance measures
	 */
	public float[] getPerformance(List<List<Integer>> truth, List<List<Integer>> pred) {
		// nHit is the number of elected alts who are also a winner
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

	public double eval(int m, int n, int s, VotingRule oracle, LearnedRule learned, Path report) {
		List<Profile<Integer>> profiles = generateProfiles(m, n, s);
		List<Integer> items = new ArrayList<>();
		for (int i = 0; i < m; i++)
			items.add(i);

		StringBuffer sb = new StringBuffer();
		sb.append("truth\tpredict\taccuracy\n");

		int total = 0;
		double accuracy = 0;
		for (Profile<Integer> profile : profiles) {
			List<Integer> winners = oracle.getAllWinners(profile);
			if (winners == null || winners.size() == 0)
				continue;

			total++;
			List<Integer> predicted = learned.getAllWinners(profile, items);
			int c = 0;
			for (int pred : predicted)
				if (winners.contains(pred))
					c++;
			double percentage = c * 1.0d / winners.size();

			accuracy += percentage;
			sb.append(winners).append("\t").append(predicted).append("\t").append(percentage).append("\n");
		}

		accuracy /= total;
		sb.append("accuracy: ").append(accuracy).append("\n");

		OpenOption[] options =
				{ StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING };

		try {
			Files.write(report, sb.toString().getBytes(), options);
		} catch (IOException e) {
			e.printStackTrace();
		}
		return accuracy;
	}

	public static void main(String[] args) throws IOException {
		TickClock.beginTick();

		int m = 10, s_train = 1000, s_eval = 1000;
		int[] votes = { 7, 9, 11 };

		List<Integer> items = new ArrayList<>();
		for (int i = 0; i < m; i++)
			items.add(i);

		VoteExp vt = new VoteExp();
		vt.fid = 2;

		List<Profile<Integer>> profiles = null;

		VotingRule oracle = new Borda();
		LearnedRule learned = null;

		StringBuffer sb = new StringBuffer();

		for (int n : votes) {
			sb.append("\n");
			profiles = vt.generateProfiles(m, n, s_train);
			learned = vt.getNNRule(profiles, items, oracle);
			// learned = vt.getMCPRule(profiles, items, oracle);

			profiles = vt.generateProfiles(m, n, s_eval);
			sb.append("Eval Perf :").append(Arrays.toString(vt.getPerformance(profiles, oracle, learned)));
		}
		System.out.println(sb.toString());

		TickClock.stopTick();
	}
}