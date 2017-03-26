package com.horsehour.vote.data;

import java.io.IOException;
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
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiFunction;
import java.util.function.Function;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.tuple.Pair;

import com.horsehour.util.MathLib;
import com.horsehour.util.TickClock;
import com.horsehour.vote.DataEngine;
import com.horsehour.vote.Profile;
import com.horsehour.vote.rule.LearnedRule;
import com.horsehour.vote.rule.VotingRule;
import com.horsehour.vote.train.VoteNetwork;

import smile.classification.NeuralNetwork;

/**
 * @author Chunheng Jiang
 * @since Mar 19, 2017
 * @version 1.0
 */
public class VoteExp2 {
	public int fid = 2;
	public double cutoff = 0.2;

	public static OpenOption[] options =
			{ StandardOpenOption.APPEND, StandardOpenOption.CREATE, StandardOpenOption.WRITE };

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

	/**
	 * Build data set from outcomes of a given voting rule on specific profiles
	 * 
	 * @param m
	 * @param base
	 * @param outcome
	 * @param data
	 * @throws IOException
	 */
	public void buildDataSet(int m, String base, Path outcome, Path data) throws IOException {
		Map<String, List<Integer>> outcomes = VoteLab.parseElectionOutcome(outcome);
		Profile<Integer> profile;
		List<Integer> truth;

		List<Integer> items = new ArrayList<>();
		for (int i = 0; i < m; i++)
			items.add(i);

		for (String name : outcomes.keySet()) {
			if (!name.contains("M" + m + "N100"))
				continue;

			profile = DataEngine.loadProfile(Paths.get(base + "/soc-3-hardcase/" + name));
			double[] features = getFeatures(profile, items);
			truth = outcomes.get(name);
			StringBuffer sb = new StringBuffer();
			sb.append(Arrays.toString(features) + "; " + truth + "\n");
			Files.write(data, sb.toString().getBytes(), options);
			System.out.println(name);
		}
	}

	Function<String[], int[]> classes = columns -> {
		int dim = columns.length;
		List<Integer> list = new ArrayList<>();
		for (int i = 0; i < dim; i++)
			if (columns[i].contains("1"))
				list.add(i);

		int[] lbls = new int[list.size()];
		for (int i = 0; i < list.size(); i++)
			lbls[i] = list.get(i);
		return lbls;
	};

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

			int[] outputs = classes.apply(columns);
			// dim = columns.length;
			// int[] outputs = new int[dim];
			// for (int k = 0; k < dim; k++)
			// outputs[k] = Integer.parseInt(columns[k]);

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

	Pair<double[][], int[]> getTrainSet(Pair<double[][], int[][]> trainset) {
		int nTotal = 0;

		for (int[] label : trainset.getRight())
			nTotal += label.length;

		double[][] inputs = new double[nTotal][];
		int[] outputs = new int[nTotal];

		double[][] in = trainset.getKey();
		int[][] out = trainset.getValue();

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
	 * K Hot Encoding
	 */
	BiFunction<Integer, int[], String> kHotEnc = (k, output) -> {
		List<Integer> list = new ArrayList<>();
		for (int i : output)
			list.add(i);
		Collections.sort(list);

		StringBuffer sb = new StringBuffer();

		for (int i = 0; i < k - 1; i++) {
			if (!list.isEmpty() && i == list.get(0))
				sb.append("1,");
			else
				sb.append("0,");
		}

		if (!list.isEmpty() && list.get(0) == k - 1)
			sb.append("1");
		else
			sb.append("0");
		return sb.toString();
	};

	/**
	 * Learned voting rule based on neural network
	 * 
	 * @param trainset
	 * @param items
	 * @param nHidden
	 * @return learned rule
	 */
	public BiFunction<double[], List<Integer>, List<Integer>> getNNRule(Pair<double[][], int[]> trainset,
			List<Integer> items, int nHidden) {
		double[][] inputs = trainset.getLeft();
		int[] output = trainset.getRight();
		int dim = inputs[0].length;
		int nItem = items.size();
		int nInput = dim, nOutput = nItem;

		// NeuralNetwork.ErrorFunction errorFunc =
		// NeuralNetwork.ErrorFunction.CROSS_ENTROPY;
		NeuralNetwork.ErrorFunction errorFunc = NeuralNetwork.ErrorFunction.LEAST_MEAN_SQUARES;

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
			// System.out.println(i + ":" + Arrays.toString(pr));

			List<Integer> pred = new ArrayList<>();
			// int[] max = MathLib.argmax(pr);
			// for (int item : max)
			// pred.add(items.get(item));

			int[] r = MathLib.getRank(pr, false);
			double sum = MathLib.Data.sum(pr);
			double theta = sum * cutoff;
			double cumu = 0;
			for (int k : r) {
				pred.add(items.get(k));
				cumu += pr[k];
				if (cumu >= theta)
					break;
			}

			preds.add(pred);
		}
		// System.out.println("Train Perf : " +
		// Arrays.toString(getPerformance(truth, preds)));

		BiFunction<double[], List<Integer>, List<Integer>> mechanism = (x, cands) -> {
			double[] pr = new double[nOutput];
			algo.predict(x, pr);
			List<Integer> pred = new ArrayList<>();
			int[] r = MathLib.getRank(pr, false);
			double sum = MathLib.Data.sum(pr);
			double theta = sum * cutoff;
			double cumu = 0;
			for (int k : r) {
				pred.add(items.get(k));
				cumu += pr[k];
				if (cumu >= theta)
					break;
			}

			// for (int i = 0; i < pr.length; i++) {
			// if (pr[i] >= 0.5)
			// pred.add(cands.get(i));
			// }
			//
			// if (pred.isEmpty()) {
			// int[] max = MathLib.argmax(pr);
			// for (int i : max)
			// pred.add(cands.get(i));
			// }
			return pred;
		};
		return mechanism;
	}

	public BiFunction<double[], List<Integer>, List<Integer>> getLogisticRule(Pair<double[][], int[]> trainset,
			List<Integer> items, int nHidden) {
		double[][] inputs = trainset.getLeft();
		int[] output = trainset.getRight();
		int dim = inputs[0].length;
		int nItem = items.size();
		int nInput = dim, nOutput = nItem;

		// NeuralNetwork.ErrorFunction errorFunc =
		// NeuralNetwork.ErrorFunction.CROSS_ENTROPY;
		NeuralNetwork.ErrorFunction errorFunc = NeuralNetwork.ErrorFunction.LEAST_MEAN_SQUARES;

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
			System.out.println(i + ":" + Arrays.toString(pr));

			List<Integer> pred = new ArrayList<>();
			int[] max = MathLib.argmax(pr);
			for (int item : max)
				pred.add(items.get(item));
			preds.add(pred);
		}
		System.out.println("Train Perf : " + Arrays.toString(getPerformance(truth, preds)));

		BiFunction<double[], List<Integer>, List<Integer>> mechanism = (x, cands) -> {
			double[] pr = new double[nOutput];
			algo.predict(x, pr);
			List<Integer> pred = new ArrayList<>();
			for (int i = 0; i < pr.length; i++) {
				if (pr[i] >= 0.5)
					pred.add(cands.get(i));
			}

			if (pred.isEmpty()) {
				int[] max = MathLib.argmax(pr);
				for (int i : max)
					pred.add(cands.get(i));
			}
			return pred;
		};
		return mechanism;
	}

	/**
	 * Learned voting rule based on neural network
	 * 
	 * @param trainset
	 * @param items
	 * @param nHidden
	 * @return learned rule
	 */
	public BiFunction<double[], List<Integer>, List<Integer>> getVoteNet(Pair<double[][], int[][]> trainset,
			List<Integer> items, int nHidden) {
		double[][] inputs = trainset.getLeft();
		int[][] output = trainset.getRight();
		int dim = inputs[0].length;
		int nItem = items.size();
		int nInput = dim, nOutput = nItem;

		// LEAST_MEAN_SQUARES, CROSS_ENTROPY
		VoteNetwork.ErrorFunction errorFunc = VoteNetwork.ErrorFunction.LEAST_MEAN_SQUARES;

		VoteNetwork.Trainer trainer = null;
		trainer = new VoteNetwork.Trainer(errorFunc, nInput, nHidden, nOutput);

		VoteNetwork algo = trainer.train(inputs, output);

		List<List<Integer>> truth = new ArrayList<>();
		List<List<Integer>> preds = new ArrayList<>();
		for (int i = 0; i < output.length; i++) {
			List<Integer> y = new ArrayList<>();
			for (int k = 0; k < output[i].length; k++)
				y.add(output[i][k]);
			truth.add(y);

			double[] input = inputs[i];
			double[] pr = new double[nOutput];
			algo.predict(input, pr);
			List<Integer> pred = new ArrayList<>();
			int[] r = MathLib.getRank(pr, false);
			double sum = MathLib.Data.sum(pr);
			double theta = sum * cutoff;
			double cumu = 0;
			for (int k : r) {
				pred.add(items.get(k));
				cumu += pr[k];
				if (cumu >= theta)
					break;
			}

			// int[] max = MathLib.argmax(pr);
			// for (int item : max)
			// pred.add(items.get(item));
			preds.add(pred);
		}
		// System.out.println("Train Perf : " +
		// Arrays.toString(getPerformance(truth, preds)));

		BiFunction<double[], List<Integer>, List<Integer>> mechanism = (x, cands) -> {
			double[] pr = new double[nOutput];
			algo.predict(x, pr);
			List<Integer> pred = new ArrayList<>();
			int[] r = MathLib.getRank(pr, false);
			double sum = MathLib.Data.sum(pr);
			double theta = sum * cutoff;
			double cumu = 0;
			for (int k : r) {
				pred.add(items.get(k));
				cumu += pr[k];
				if (cumu >= theta)
					break;
			}

			// int[] max = MathLib.argmax(pr);
			// for (int i : max)
			// pred.add(cands.get(i));
			return pred;
		};
		return mechanism;
	}

	public Function<String, Pair<double[], int[]>> extractor = line -> {
		String[] fields = line.split("\t|;");
		String[] columns = fields[0].trim().replaceAll("\\[|\\]| ", "").split(",");
		int dim = columns.length;
		double[] inputs = new double[dim];
		for (int k = 0; k < dim; k++)
			inputs[k] = Double.parseDouble(columns[k].trim());
		columns = fields[1].replaceAll("\\[|\\]| ", "").split(",");
		dim = columns.length;
		int[] outputs = new int[dim];
		for (int k = 0; k < dim; k++)
			outputs[k] = Integer.parseInt(columns[k].trim());
		return Pair.of(inputs, outputs);
	};

	/**
	 * Convert data set generated by Jun to SVM or standard k-encoded
	 * 
	 * @param sourcePath
	 * @param sinkPath
	 * @param svm
	 * @throws IOException
	 */
	public void convert(Path sourcePath, Path sinkPath, boolean svm) throws IOException {
		if (svm)
			Files.lines(sourcePath).forEach(line -> {
				Pair<double[], int[]> entry = extractor.apply(line);
				StringBuffer sb = new StringBuffer();
				int[] khot = entry.getRight();
				List<Integer> outputs = new ArrayList<>();
				for (int i = 0; i < khot.length; i++) {
					if (khot[i] == 1)
						outputs.add(i);
				}
				String string = outputs.toString();
				string = string.replaceAll("\\[|\\]| ", "");
				sb.append(string);
				double[] inputs = entry.getKey();
				for (int i = 0; i < inputs.length; i++)
					sb.append(" ").append(i + 1).append(":").append(inputs[i]);
				sb.append("\n");
				try {
					Files.write(sinkPath, sb.toString().getBytes(), options);
				} catch (IOException e) {
					e.printStackTrace();
				}
			});
		else
			Files.lines(sourcePath).forEach(line -> {
				Pair<double[], int[]> entry = extractor.apply(line);
				StringBuffer sb = new StringBuffer();
				String string = Arrays.toString(entry.getLeft());
				string = string.replaceAll("\\[|\\]| ", "");
				sb.append(string).append("; ");

				String coding = Arrays.toString(entry.getRight());
				coding = coding.replaceAll("\\[|\\]| ", "");
				sb.append(coding).append("\n");
				try {
					Files.write(sinkPath, sb.toString().getBytes(), options);
				} catch (IOException e) {
					e.printStackTrace();
				}
			});
	}

	public void convert(Path sourcePath, Path sinkPath, int m, boolean svm) throws IOException {
		if (svm)
			Files.lines(sourcePath).forEach(line -> {
				Pair<double[], int[]> entry = extractor.apply(line);
				StringBuffer sb = new StringBuffer();
				String string = Arrays.toString(entry.getRight());
				string = string.replaceAll("\\[|\\]| ", "");
				sb.append(string);
				double[] inputs = entry.getKey();
				for (int i = 0; i < inputs.length; i++)
					sb.append(" ").append(i + 1).append(":").append(inputs[i]);
				sb.append("\n");
				try {
					Files.write(sinkPath, sb.toString().getBytes(), options);
				} catch (IOException e) {
					e.printStackTrace();
				}
			});
		else
			Files.lines(sourcePath).forEach(line -> {
				Pair<double[], int[]> entry = extractor.apply(line);
				StringBuffer sb = new StringBuffer();
				String string = Arrays.toString(entry.getLeft());
				string = string.replaceAll("\\[|\\]| ", "");
				sb.append(string).append("; ");

				String coding = kHotEnc.apply(m, entry.getRight());
				sb.append(coding).append("\n");
				try {
					Files.write(sinkPath, sb.toString().getBytes(), options);
				} catch (IOException e) {
					e.printStackTrace();
				}
			});
	}

	public void convert(Pair<double[][], int[][]> source, Path sinkPath, int m, boolean svm) throws IOException {
		double[][] inputs = source.getKey();
		int[][] outputs = source.getValue();
		int n = inputs.length;
		StringBuffer sb = null;
		if (svm)
			for (int i = 0; i < n; i++) {
				double[] input = inputs[i];
				int[] output = outputs[i];
				sb = new StringBuffer();
				String string = Arrays.toString(output);
				string = string.replaceAll("\\[|\\]| ", "");
				sb.append(string);
				for (int k = 0; k < input.length; k++)
					sb.append(" ").append(k + 1).append(":").append(input[k]);
				sb.append("\n");
				Files.write(sinkPath, sb.toString().getBytes(), options);
			}
		else {
			for (int i = 0; i < n; i++) {
				double[] input = inputs[i];
				int[] output = outputs[i];
				sb = new StringBuffer();
				String string = Arrays.toString(input);
				string = string.replaceAll("\\[|\\]| ", "");
				sb.append(string).append("; ");
				String coding = kHotEnc.apply(m, output);
				sb.append(coding).append("\n");
				Files.write(sinkPath, sb.toString().getBytes(), options);
			}
		}
	}

	public static void main_0(String[] args) throws IOException {
		TickClock.beginTick();

		VoteExp2 tr1 = new VoteExp2();
		tr1.fid = 2;

		String base = "/users/chjiang/documents/csc/";
		Path data = Paths.get(base + "/jun-feature3.csv");
		tr1.convert(data, Paths.get(base + "/jun-soc-3-m10n100.csv"), false);

		// List<Pair<double[][], int[][]>> dataset = new ArrayList<>();
		// tr1.split(data, dataset, 0.7F);
		// Path sinkPath = Paths.get(base + "/jun-soc-3-m10n100-train.csv");
		// int m = 10;
		// tr1.convert(dataset.get(0), sinkPath, m, false);
		// sinkPath = Paths.get(base + "/jun-soc-3-m10n100-test.csv");
		// tr1.convert(dataset.get(1), sinkPath, m, false);

		TickClock.stopTick();
	}

	public static void main_2(String[] args) throws IOException {
		String base = "/users/chjiang/documents/csc/learn/jun-soc-3-m10n100/";
		// Path data = Paths.get(base + "/data.csv");
		Path sink = Paths.get(base + "/data.svm");

		split(sink, Paths.get(base + "/train.svm"), Paths.get(base + "/test.svm"), 0.7F);

		// StringBuffer sb = new StringBuffer();
		// List<String> lines = Files.readAllLines(data);
		// for (String line : lines) {
		// sb.append(svm.apply(line));
		// }
		// Files.write(sink, sb.toString().getBytes(), options);
	}

	public static void split(Path input, Path train, Path eval, float ratioTrain) throws IOException {
		int n = (int) (Files.lines(input).count());
		int m = (int) (n * ratioTrain);

		List<Integer> trainList = MathLib.Rand.sample(0, n, m);
		Collections.sort(trainList);

		AtomicInteger ind = new AtomicInteger(0);
		AtomicInteger nTrain = new AtomicInteger(0);
		Files.lines(input).forEach(line -> {
			line += "\n";
			try {
				int i = nTrain.get();
				if (i < m && trainList.get(i) == ind.get()) {
					Files.write(train, line.getBytes(), options);
					nTrain.getAndIncrement();
				} else
					Files.write(eval, line.getBytes(), options);
			} catch (IOException e) {
				e.printStackTrace();
			}
			ind.getAndIncrement();
		});
	}

	static Function<String, String> svm = line -> {
		String[] cat = line.split(";");
		StringBuffer sb = new StringBuffer();
		String[] labels = cat[1].trim().split(",");
		List<Integer> labelList = new ArrayList<>();
		for (int i = 0; i < labels.length; i++) {
			if (labels[i].contains("1")) {
				labelList.add(i);
			}
		}
		String label = labelList.toString().replaceAll("\\[|\\]| ", "");
		sb.append(label).append(" ");

		String[] features = cat[0].trim().split(",");
		for (int i = 0; i < features.length; i++) {
			sb.append(i + 1).append(":").append(features[i]);
			if (i < features.length - 1)
				sb.append(" ");
			else
				sb.append("\n");
		}
		return sb.toString();
	};

	public static void main(String[] args) throws IOException {
		TickClock.beginTick();

		int m = 10;

		VoteExp2 tr1 = new VoteExp2();
		tr1.fid = 2;

		String base = "/users/chjiang/documents/csc/";
		Path data = Paths.get(base + "/M10.csv");

		for (int iter = 1; iter <= 1; iter++) {
			List<Pair<double[][], int[][]>> dataset = new ArrayList<>();
			tr1.split(data, dataset, 0.7F);

			Pair<double[][], int[]> samples = tr1.getTrainSet(dataset.get(0));
			List<Integer> items = new ArrayList<>();
			for (int i = 0; i < m; i++)
				items.add(i);

			BiFunction<double[], List<Integer>, List<Integer>> function = null;

			int nHidden = m;
			function = tr1.getNNRule(samples, items, nHidden);
			// function = tr1.getVoteNet(dataset.get(0), items, nHidden);

			double[][] inputs = dataset.get(1).getKey();
			int[][] outputs = dataset.get(1).getValue();

			List<List<Integer>> truth = new ArrayList<>();
			List<List<Integer>> pred = new ArrayList<>();

			for (int i = 0; i < inputs.length; i++) {
				double[] input = inputs[i];
				List<Integer> winners = new ArrayList<>();
				for (int k = 0; k < outputs[i].length; k++)
					winners.add(outputs[i][k]);

				truth.add(winners);
				pred.add(function.apply(input, items));
			}

			float[] perf = tr1.getPerformance(truth, pred);
			System.out.println(Arrays.toString(perf));
		}

		// 1. vc-dimension of general scoring voting rule (complexity to
		// evaluate whether it's an unique winner
		// 2. stv - complexity, preliminary results
		// 3. directly tell the number of winners, predict the unique winner,
		// 50% unique and 50% two winners
		// 4. reward of removing an alternative from the state can be the steps
		// required to find the maximum possible number of winners, evaluate the
		// value using some sampling

		TickClock.stopTick();
	}
}