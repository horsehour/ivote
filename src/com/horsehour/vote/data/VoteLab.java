package com.horsehour.vote.data;

import java.io.IOException;
import java.nio.file.CopyOption;
import java.nio.file.Files;
import java.nio.file.OpenOption;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.function.Function;
import java.util.stream.Collectors;

import org.apache.commons.lang3.tuple.Pair;

import com.horsehour.ml.data.Data;
import com.horsehour.ml.data.SampleSet;
import com.horsehour.util.MathLib;
import com.horsehour.util.TickClock;
import com.horsehour.vote.DataEngine;
import com.horsehour.vote.Profile;
import com.horsehour.vote.rule.LearnedRule;
import com.horsehour.vote.rule.multiseat.Baldwin;
import com.horsehour.vote.rule.multiseat.Coombs;
import com.horsehour.vote.rule.multiseat.STVPlus2;
import com.horsehour.vote.rule.multiseat.VoteResearch;

import smile.classification.LogisticRegression;

/**
 *
 * @author Chunheng Jiang
 * @version 1.0
 * @since 10:54:14 PM, Feb 15, 2017
 *
 */
public class VoteLab {
	static OpenOption[] options = { StandardOpenOption.APPEND, StandardOpenOption.CREATE, StandardOpenOption.WRITE };
	static DateTimeFormatter fmt = DateTimeFormatter.ofPattern("yyyyMMdd HH:mm:ss.SSSSSS");

	/**
	 * Experiment contains detailed information about a voting rule with various
	 * settings, including the running time, number of times to invoke the
	 * scoring function, number of nodes have been extended
	 * 
	 * @param method
	 * @param base
	 * @param dataset
	 * @param ms
	 * @param ns
	 * @param k
	 * @param heuristic
	 * @param cache
	 * @param pruning
	 * @param sampling
	 * @param queue
	 * @param recursive
	 * @throws IOException
	 */
	public static void experiment(String method, String base, String dataset, int[] ms, int[] ns, int[] k,
			boolean heuristic, boolean cache, boolean pruning, boolean sampling, boolean recursive, int pFunction)
			throws IOException {
		if (ms == null || ns == null) {
			experiment(method, base, dataset, -1, -1, null, heuristic, cache, pruning, sampling, recursive, pFunction);
			return;
		}

		// soc-3
		for (int m = ms[0]; m <= ms[1]; m++)
			for (int n = ns[0]; n <= ns[1]; n++) {
				experiment(method, base, dataset, m, n, k, heuristic, cache, pruning, sampling, recursive, pFunction);
			}
	}

	public static void experiment(String method, String base, String dataset, int m, int n, int[] k, boolean heuristic,
			boolean cache, boolean pruning, boolean sampling, boolean recursive, int pFunction) throws IOException {

		STVPlus2 rule = null;
		if (method.contains("stv")) {
			rule = new STVPlus2(heuristic, cache, pruning, sampling, recursive, pFunction);
		} else if (method.contains("coombs")) {
			rule = new Coombs(heuristic, cache, pruning, sampling, recursive, pFunction);
		} else if (method.contains("baldwin")) {
			rule = new Baldwin(heuristic, cache, pruning, sampling, recursive, pFunction);
		}

		String hp = "";
		if (k == null)
			hp = base + dataset + "-" + method;
		else
			hp = base + dataset + "-" + method + "-k" + k[0] + "-" + k[1];

		int h = heuristic ? 1 : 0, c = cache ? 1 : 0, p = pruning ? 1 : 0, s = sampling ? 1 : 0, r = recursive ? 1 : 0;
		hp += "-h" + h + "c" + c + "p" + p + "s" + s + "r" + r + "pf" + pFunction + ".txt";

		Path output = Paths.get(hp);
		if (m <= 0 || n <= 0) {
			List<Path> files = Files.list(Paths.get(base + dataset)).collect(Collectors.toList());
			for (Path file : files) {
				String name = file.toFile().getName();
				Profile<Integer> profile = DataEngine.loadProfile(file);
				List<Integer> winners = rule.getAllWinners(profile);

				System.out.println(name + "\t" + ZonedDateTime.now().format(fmt));
				report(rule, winners, name, sampling, output);
			}
			return;
		}

		Path file = null;
		Profile<Integer> profile;
		int count = 0;

		for (int i = 1; i <= 2000; i++) {
			String name = "M" + m + "N" + n + "-" + i + ".csv";
			file = Paths.get(base + dataset + "-hardcase/" + name);
			if (Files.exists(file)) {
				count++;
				if (k != null) {
					if (count < k[0])
						continue;
					else if (count > k[1])
						return;
				}

				System.out.println(name + "\t" + ZonedDateTime.now().format(fmt));

				profile = DataEngine.loadProfile(file);
				List<Integer> winners = rule.getAllWinners(profile);
				report(rule, winners, name, sampling, output);
			}
		}
	}

	/**
	 * report experimental results
	 * 
	 * @param rule
	 * @param winners
	 * @param name
	 * @param sampling
	 * @param output
	 * @throws IOException
	 */
	static void report(STVPlus2 rule, List<Integer> winners, String name, boolean sampling, Path output)
			throws IOException {
		String format = "%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%s\t%s\t%s";
		float ratio = rule.timeScoring + rule.timeCacheEval + rule.timeHeuristicEval + rule.timeFork
				+ rule.timeSelectNext + rule.timePruneEval + rule.timePred + rule.timeComputePriority;

		if (rule.time == 0)
			ratio = 1.0F;
		else
			ratio /= rule.time;

		StringBuffer sb = new StringBuffer();
		if (sampling) {
			sb.append(String.format(format + "\t%s\n", name, rule.numItemTotal, rule.numFailH, rule.numSingleL,
					rule.numMultiL, rule.numScoring, rule.numNodeWH, rule.numNodeWOH, rule.numNode, rule.numNodeFull,
					rule.numCacheHit, rule.numCacheMiss, rule.numPruneHit, rule.numPruneMiss, rule.time,
					rule.timeScoring, rule.timeHeuristicEval, rule.timeCacheEval, rule.timePruneEval, rule.timeSampling,
					rule.timeSelectNext, rule.timeFork, rule.timePred, rule.timeComputePriority, ratio, winners,
					rule.freq, rule.trace, rule.electedSampling));
		} else {
			sb.append(String.format(format + "\n", name, rule.numItemTotal, rule.numFailH, rule.numSingleL,
					rule.numMultiL, rule.numScoring, rule.numNodeWH, rule.numNodeWOH, rule.numNode, rule.numNodeFull,
					rule.numCacheHit, rule.numCacheMiss, rule.numPruneHit, rule.numPruneMiss, rule.time,
					rule.timeScoring, rule.timeHeuristicEval, rule.timeCacheEval, rule.timePruneEval, rule.timeSampling,
					rule.timeSelectNext, rule.timeFork, rule.timePred, rule.timeComputePriority, ratio, winners,
					rule.freq, rule.trace));
		}
		Files.write(output, sb.toString().getBytes(), options);
	}

	/**
	 * Based on the hardness to apply the heurstic search algorithm (HSA) to
	 * elect a STV winner, each profile falls into one of the following three
	 * hardness levels:
	 * <li>1. extremely easy case - there is no ties in the profile</li>
	 * <li>2. easy case: produce a single winner using only HSA</li>
	 * <li>3. hard case: HSA can not process at some step</li>
	 */

	public static boolean tied = false;

	/**
	 * compute the hardness of a given profile
	 * 
	 * @param rule
	 * @param profile
	 * @return hardness of a profile
	 */
	public static int getHardness(String rule, Profile<Integer> profile) {
		List<Integer> state = new ArrayList<>(Arrays.asList(profile.getSortedItems()));
		String rname = rule.toLowerCase();
		if (rname.startsWith("stv")) {
			int[] scores = plurality(profile, state);
			List<Integer> losers = null;
			TreeMap<Integer, List<Integer>> tiers = new TreeMap<>(Collections.reverseOrder());
			while ((losers = getHeuristicLoser(state, scores, tiers)) != null && state.size() > 1) {
				state.removeAll(losers);
				scores = plurality(profile, state);
			}

			// no tie
			if (!tied)
				return 1;

			if (state.size() > 1)
				return 3;
			else
				return 2;
		}

		if (rname.startsWith("baldwin")) {
			getPrefMatrix(profile);
			int[] min = null;
			while (state.size() > 1) {
				int[] scores = borda(profile, state);
				min = MathLib.argmin(scores);
				if (min.length > 1)
					return 2;
				state.remove(state.get(min[0]));
			}
			return 1;
		}

		if (rname.startsWith("coombs")) {
			int[] min = null;
			while (state.size() > 1) {
				int[] scores = veto(profile, state);
				min = MathLib.argmin(scores);
				if (min.length > 1)
					return 2;
				state.remove(state.get(min[0]));
			}
			return 1;
		}

		// undefined
		return -1;
	}

	static Map<Integer, Map<Integer, Integer>> prefMatrix;

	static void getPrefMatrix(Profile<Integer> profile) {
		prefMatrix = new HashMap<>();
		int[] votes = profile.votes;
		int m = profile.getNumItem();
		for (int k = 0; k < votes.length; k++) {
			Integer[] pref = profile.data[k];
			for (int i = 0; i < m; i++) {
				Map<Integer, Integer> outlink = prefMatrix.get(pref[i]);
				if (outlink == null) {
					outlink = new HashMap<>();
					prefMatrix.put(pref[i], outlink);
				}

				for (int j = i + 1; j < m; j++) {
					if (outlink.get(pref[j]) == null)
						outlink.put(pref[j], votes[k]);
					else
						outlink.put(pref[j], outlink.get(pref[j]) + votes[k]);
				}
			}
		}
	}

	/**
	 * Calculate the borda scores of the remaining alternatives in the current
	 * state
	 * 
	 * @param profile
	 * @param state
	 * @return borda scores
	 */
	public static int[] borda(Profile<Integer> profile, List<Integer> state) {
		int m = state.size();
		int[] scores = new int[m];

		Map<Integer, Integer> outlink;
		for (int i = 0; i < m; i++) {
			int item1 = state.get(i);
			outlink = prefMatrix.get(item1);
			for (int item2 : state) {
				if (item2 == item1 || outlink.get(item2) == null)
					continue;
				scores[i] += outlink.get(item2);
			}
		}
		return scores;
	}

	/**
	 * Calculate the plurality scores of the remaining alternatives in the
	 * current state
	 * 
	 * @param profile
	 * @param state
	 * @return plurality scores
	 */
	public static int[] plurality(Profile<Integer> profile, List<Integer> state) {
		int[] votes = profile.votes;
		int[] scores = new int[state.size()];

		int c = 0;
		for (Integer[] pref : profile.data) {
			for (int i = 0; i < pref.length; i++) {
				int item = pref[i];
				int index = state.indexOf(item);
				// item has been eliminated
				if (index == -1)
					continue;
				scores[index] += votes[c];
				break;
			}
			c++;
		}
		return scores;
	}

	/**
	 * Calculate the veto scores of the remaining alternatives in the current
	 * state
	 * 
	 * @param profile
	 * @param state
	 * @return vote scores
	 */
	public static int[] veto(Profile<Integer> profile, List<Integer> state) {
		int[] votes = profile.votes;
		int[] scores = new int[state.size()];

		int c = 0;
		for (Integer[] pref : profile.data) {
			for (int i = pref.length - 1; i >= 0; i--) {
				int item = pref[i];
				int index = state.indexOf(item);
				// item has been eliminated
				if (index == -1)
					continue;
				scores[index] += votes[c];
				break;
			}
			c++;
		}
		return scores;
	}

	/**
	 * According to the constraints on votes, we get all the candidates who will
	 * be eliminated no matter which tie-breaking rule to be used.
	 * 
	 * @param items
	 * @param scores
	 * @param tiers
	 * @return eliminated candidates
	 */
	static List<Integer> getHeuristicLoser(List<Integer> items, int[] scores, TreeMap<Integer, List<Integer>> tiers) {
		tiers.clear();
		for (int i = 0; i < scores.length; i++) {
			int score = scores[i];
			List<Integer> member = null;
			if (tiers.containsKey(score)) {
				tied = true;
				member = tiers.get(score);
				member.add(items.get(i));
			} else {
				member = new ArrayList<>();
				member.add(items.get(i));
			}
			tiers.put(score, member);
		}

		// all candidates have the same score
		if (tiers.size() == 1)
			return null;

		List<Integer> distinct = new ArrayList<>(tiers.keySet());
		int i = 0;
		int remaining = MathLib.Data.sum(scores);
		for (; i < distinct.size(); i++) {
			int score = distinct.get(i);
			int scoreLocal = score * tiers.get(score).size();
			remaining -= scoreLocal;
			if (remaining < score)
				break;
		}

		if (i == distinct.size() - 1)
			return null;

		i += 1;
		List<Integer> losers = new ArrayList<>();
		for (; i < distinct.size(); i++) {
			int score = distinct.get(i);
			losers.addAll(tiers.get(score));
		}
		return losers;
	}

	public static void getHardCases(String rule, String base, String dataset) throws IOException {
		Path output = Paths.get(base + dataset + "-hardness.txt");
		List<Path> files = Files.list(Paths.get(base + dataset + "/")).collect(Collectors.toList());
		CopyOption[] cpOptions = { StandardCopyOption.COPY_ATTRIBUTES, StandardCopyOption.REPLACE_EXISTING };

		String rname = rule.toLowerCase();
		for (Path file : files) {
			Profile<Integer> profile = DataEngine.loadProfile(file);
			int hardness = getHardness(rule, profile);
			String name = file.toFile().getName();

			String content = name + "\t" + hardness + "\n";
			Files.write(output, content.getBytes(), options);
			// 3-level hardness for STV, and 2-level for other methods
			if (hardness == 3 || (!rname.startsWith("stv") && hardness == 2)) {
				System.out.println(name);
				Files.copy(Paths.get(base + dataset + "/" + name), Paths.get(base + dataset + "-hardcase/" + name),
						cpOptions);
			}
		}
	}

	/**
	 * Measure the hardness of profiles and record it on local machine
	 * 
	 * @param rule
	 * @param files
	 * @param output
	 * @throws IOException
	 */
	public static void hardness(String rule, List<Path> files, Path output) throws IOException {
		Profile<Integer> profile;
		for (Path file : files) {
			profile = DataEngine.loadProfile(file);
			int hardness = getHardness(rule, profile);
			String name = file.toFile().getName();

			System.out.println(name);

			String line = name + "\t" + hardness + "\n";
			Files.write(output, line.getBytes(), options);
		}
	}

	/**
	 * Experiment on the proability distribution of number of hard cases w.r.t
	 * different m the number of alternatives, n the number of voters by the
	 * evaluation on randomly generated k (for each pair of m and n) profiles
	 * 
	 * @param rule
	 * @param ms
	 * @param ns
	 * @param k
	 */
	public static void getRatioHardCase(String rule, int[] ms, int[] ns, int k) {
		Profile<Integer> profile;
		System.out.print("m");
		for (int n : ns)
			System.out.print("\t" + n);
		System.out.println("");

		String rname = rule.toLowerCase();
		for (int m : ms) {
			System.out.print(m);
			for (int n : ns) {
				int count = 0;
				for (int i = 0; i < k; i++) {
					profile = DataEngine.getRandomProfile(m, n);
					if (rname.startsWith("stv")) {
						if (getHardness(rule, profile) == 3)
							count++;
					} else if (getHardness(rule, profile) == 2)
						count++;
				}
				System.out.print("\t" + count * 1.0d / k + "(" + count + ")");
			}
			System.out.println("\r\n");
		}
	}

	public static class Efficiency {
		static void recordTrace(boolean heuristic, boolean cache, boolean pruning, boolean sampling, boolean recursive,
				int pFunction) throws IOException {

			String base = "/Users/chjiang/Documents/csc/";
			String dataset = "soc-3";
			STVPlus2 rule = new STVPlus2(heuristic, cache, pruning, sampling, recursive, pFunction);
			String hp = base + dataset + "-k1-1000";
			int h = heuristic ? 1 : 0, c = cache ? 1 : 0, p = pruning ? 1 : 0, s = sampling ? 1 : 0,
					r = recursive ? 1 : 0;
			hp += "-h" + h + "c" + c + "p" + p + "s" + s + "r" + r + "pf" + pFunction + "-trace.txt";

			Path output = Paths.get(hp);

			DateTimeFormatter fmt = DateTimeFormatter.ofPattern("yyyyMMdd HH:mm:ss.SSSSSS");
			OpenOption[] options = { StandardOpenOption.APPEND, StandardOpenOption.CREATE, StandardOpenOption.WRITE };

			Path file;
			Profile<Integer> profile;

			StringBuffer sb = new StringBuffer();

			for (int m = 10; m <= 30; m += 10) {
				for (int n = 10; n <= 100; n += 10) {
					int count = 0;
					for (int k = 1; k <= 1500; k++) {
						String name = "M" + m + "N" + n + "-" + k + ".csv";
						// file = Paths.get(base + dataset + "-hardcase/" +
						// name);
						file = Paths.get(base + dataset + "/" + name);
						if (Files.exists(file))
							count++;
						else
							continue;

						if (count > 1000)
							break;

						System.out.println(name + "\t" + ZonedDateTime.now().format(fmt));

						sb.append(m + "\t" + n + "\t" + k);
						profile = DataEngine.loadProfile(file);
						List<Integer> winners = rule.getAllWinners(profile);
						int yTotal = winners.size();
						int xTotal = rule.numNode;
						for (int x : rule.trace.keySet()) {
							int y = rule.trace.get(x);
							sb.append("\t" + x * 1.0d / xTotal + ":" + y * 1.0d / yTotal);
						}
						sb.append("\r\n");

						Files.write(output, sb.toString().getBytes(), options);
						sb = new StringBuffer();
					}

				}
			}
		}

		static void extractTrace(Path input, Path output) throws IOException {
			int num = 100;
			String meta = "m,n,k";
			for (int i = 0; i <= num; i++) {
				meta += "," + i;
			}
			meta += "\r\n";

			Files.write(output, meta.getBytes(), options);

			Files.lines(input).forEach(line -> {
				String[] fields = line.split("\t");
				StringBuffer sb = new StringBuffer();
				sb.append(fields[0] + ",").append(fields[1] + ",").append(fields[2]);

				System.out.println(sb.toString());

				int n = fields.length - 3;
				double d = 0.01;
				if (n <= num) {
					int k = 0;
					String winner = "";
					for (int i = 3; i < fields.length; i++) {
						String[] pair = fields[i].split(":");
						double numNode = Double.parseDouble(pair[0]);
						if (i == 3) {
							winner = pair[1];
							sb.append(",").append(winner);
							k++;
							continue;
						}

						while (k * d < numNode) {
							sb.append(",").append(winner);
							k++;
						}
						winner = pair[1];
					}
					if (k * d == 1)
						sb.append(",").append(winner);
				} else {
					int k = 0;
					String winner = "";
					for (int i = 3; i < fields.length; i++) {
						String[] pair = fields[i].split(":");
						double numNode = Double.parseDouble(pair[0]);
						if (i == 3) {
							winner = pair[1];
							sb.append(",").append(winner);
							k++;
							continue;
						}

						if (numNode >= k * d) {
							sb.append(",").append(winner);
							winner = pair[1];
							k++;
						}
					}
				}

				sb.append("\r\n");
				try {
					Files.write(output, sb.toString().getBytes(), options);
				} catch (IOException e) {
					e.printStackTrace();
				}
			});
		}
	}

	/**
	 * Check whether different methods produces the same result
	 * 
	 * @param files
	 * @throws IOException
	 */
	public void consistent(List<Path> files) throws IOException {
		List<Path> paths = new ArrayList<>();
		for (Path file : files)
			paths.add(file);

		Map<String, Set<String>> results = new HashMap<>();
		for (Path path : paths) {
			for (String line : Files.readAllLines(path)) {
				int idx = line.indexOf("\t");
				String profile = line.substring(0, idx);
				idx = profile.indexOf(".");
				if (idx > -1)
					profile = profile.substring(0, idx);

				Set<String> values = results.get(profile);
				if (values == null)
					values = new HashSet<>();

				idx = line.indexOf("[");
				String winner = line.substring(idx + 1, line.indexOf("]"));
				values.add(winner);
				results.put(profile, values);
			}
		}

		for (String profile : results.keySet()) {
			if (results.get(profile).size() > 1)
				System.out.println(profile);
		}
	}

	public static class LearnVotingRule {
		/**
		 * Create data set from known election records
		 * 
		 * @param base
		 *            place that the specific data set is housed (name of data
		 *            set, and the absolute path)
		 * @param outcome
		 *            election outcomes, including the profile name, the winners
		 *            and other data
		 * @param dataFile
		 *            destination to keep all the data set.
		 * @param nSample
		 *            number of profiles used to create data set. If nSample <=
		 *            0, all profiles will be used to create the data set
		 * @throws IOException
		 */
		static void getDataSet(String base, Path outcome, Path dataFile, int nSample) throws IOException {
			List<String> lines = null;
			try {
				lines = Files.readAllLines(outcome);
			} catch (IOException e) {
				e.printStackTrace();
				return;
			}

			List<Integer> index = null;
			if (nSample > 0)
				// sampling
				index = MathLib.Rand.sample(0, lines.size(), nSample);
			else {
				index = new ArrayList<>();
				for (int i = 0; i < lines.size(); i++)
					index.add(i);
			}

			Profile<Integer> profile;

			for (int i : index) {
				String line = lines.get(i);
				Pair<String, List<Integer>> pair = parseElection(line);
				String name = pair.getKey();
				System.out.println(name);

				List<Integer> winners = pair.getValue();

				profile = DataEngine.loadProfile(Paths.get(base + name));
				double[][] features = DataEngine.getFeatures(profile, Arrays.asList(profile.getSortedItems()));

				StringBuffer sb = new StringBuffer();
				for (int k = 0; k < features.length; k++) {
					String content = "";
					for (double feature : features[k])
						content += feature + ",";

					if (winners.contains(k))
						content += "1";
					else
						content += "0";
					sb.append(content).append("\r\n");
				}
				Files.write(dataFile, sb.toString().getBytes(), options);
			}
		}
	}

	/**
	 * Create training set from known election outcomes
	 * 
	 * @param base
	 * @param outcome
	 * 
	 * @throws IOException
	 */
	static List<Pair<double[], Integer>> getTrainSet(String base, Path outcome) throws IOException {
		Map<String, List<Integer>> election = parseElectionOutcome(outcome);
		Path profileFile;
		Profile<Integer> profile;

		List<Pair<double[], Integer>> trainset = new ArrayList<>();
		for (String name : election.keySet()) {
			profileFile = Paths.get(base + name);
			profile = DataEngine.loadProfile(profileFile);
			double[][] features = DataEngine.getFeatures(profile, Arrays.asList(profile.getSortedItems()));
			List<Integer> winners = election.get(name);
			for (int i = 0; i < features.length; i++) {
				if (winners.contains(i))
					trainset.add(Pair.of(features[i], 1));
				else
					trainset.add(Pair.of(features[i], 0));
			}
		}
		return trainset;
	}

	/**
	 * Extract election results that kept on the local machine
	 * 
	 * @param outcome
	 * @return map with profile file's name as the key and the corresponding
	 *         winners are the values
	 * @throws IOException
	 */
	public static Map<String, List<Integer>> parseElectionOutcome(Path outcome) throws IOException {
		Map<String, List<Integer>> map = new HashMap<>();
		List<String> lines = Files.readAllLines(outcome);
		for (String line : lines) {
			Pair<String, List<Integer>> pair = parseElection(line);
			map.put(pair.getKey(), pair.getValue());
		}
		return map;
	}

	/**
	 * Make prediction using the learned algorithm
	 * 
	 * @param profile
	 * @param items
	 * @return prediction score of being selected for all the alternatives
	 */
	static List<Double> predict(Profile<Integer> profile, List<Integer> items) {
		double[][] features = DataEngine.getFeatures(profile, items);
		List<Double> p = new ArrayList<>();
		for (double[] x : features)
			p.add(predict(x));
		return p;
	}

	static double[][] w = { { 1.0103063484121213, 0.9696350124299993, 1.0006123390136652, 0 },
			{ 0.9896936516037581, 1.0303649875797136, 0.999387660985303, 0 } };
	static double[] b = { 0.012000000569969416, -0.012000000569969416 };

	/**
	 * @param x
	 * @return probability of being a winner
	 */
	static double predict(double[] x) {
		double[] predict = new double[2];
		for (int c = 0; c < 2; c++)
			predict[c] = MathLib.Matrix.dotProd(w[c], x) + b[c];
		return 1.0d / (1 + Math.exp(predict[0] - predict[1]));
	}

	static Pair<String, List<Integer>> parseElection(String line) {
		int ind = line.indexOf("\t");
		String id = line.substring(0, ind);
		ind = line.indexOf("[");
		String val = line.substring(ind + 1, line.indexOf("]", ind)).trim();
		String[] cells = val.split(",");
		int m = cells.length;
		List<Integer> winners = new ArrayList<>();
		for (int i = 0; i < m; i++) {
			String cell = cells[i].trim();
			int winner = Integer.parseInt(cell);
			winners.add(winner);
		}
		return Pair.of(id, winners);
	}

	/**
	 * Evaluation the performance of the predictor
	 * 
	 * @param outcome
	 *            voting outcome from a specific voting rule
	 * @throws IOException
	 */
	public static void predictionEval(Path outcome) throws IOException {
		String base = "/users/chjiang/documents/csc/", dataset = "soc-3";
		STVPlus2 rule = new STVPlus2();

		StringBuffer sb = new StringBuffer();
		sb.append("profile\tnum_winners\ttp\ttn\tfp\tfn\r\n");
		Profile<Integer> profile;
		List<String> lines = Files.readAllLines(outcome);
		double accuracy = 0;
		for (String line : lines) {
			Pair<String, List<Integer>> cont = parseElection(line);
			String name = cont.getKey();
			Path input = Paths.get(base + dataset + "/" + name);
			if (!Files.exists(input))
				input = Paths.get(base + dataset + "-hardcase/" + name);
			profile = DataEngine.loadProfile(input);

			Integer[] items = profile.getSortedItems();
			List<Integer> state = new ArrayList<>(Arrays.asList(items));
			profile = rule.preprocess(profile, state);
			List<Double> pred = predict(profile, state);
			List<Integer> winners = cont.getRight();
			BitSet truth = new BitSet(pred.size());
			for (int winner : winners)
				truth.set(state.indexOf(winner));

			double correct = 0;
			for (int i = 0; i < pred.size(); i++) {
				double p = pred.get(i);
				if ((truth.get(i) && p > 0.5) || (!truth.get(i) && p <= 0.5))
					correct++;
			}
			accuracy += (correct / pred.size());
		}
		accuracy /= lines.size();
		System.out.println(accuracy);
	}

	/**
	 * Suppose there are 3 winners, we have a notation [1=6, 2=7, 3=13]
	 * indicates that the number of nodes required to visit to catch 1, 2 and
	 * all 3 winners. Here, the method visits 6 nodes for 1 winner, and visits
	 * another 1 node to get 2 winners. Finally, it has to visit 13 nodes to
	 * elect all 3 winners. Let it be the outcome of the standard DFS method.
	 * Another proposed method have the outcome [1=4, 2=6, 3=10], we say the
	 * proposed method is more efficient than DFS to search all the winners.
	 * Therefore, we will compute the average percentage of number of nodes that
	 * the proposed method can reduced to catch 1 winner, m winners (if
	 * available).
	 * 
	 * @param baseline
	 * @param proposed
	 * @param improved
	 * @throws IOException
	 */
	public static void reducedNumNode(Path baseline, Path proposed, Path improved) throws IOException {
		List<String> base = Files.readAllLines(baseline);
		List<String> prop = Files.readAllLines(proposed);

		Function<String, Pair<String, int[]>> parser = line -> {
			int ind = line.indexOf("\t");
			String id = line.substring(0, ind).replace(".csv", "");
			ind = line.indexOf("{");
			ind = line.indexOf("{", ind + 1);
			String val = line.substring(ind + 1, line.indexOf("}", ind)).trim();
			String[] cells = val.split(",");
			int m = cells.length;
			int[] numNodes = new int[m - 1];
			for (int i = 1; i < m; i++) {
				String cell = cells[i];
				String element = cell.substring(cell.indexOf("=") + 1);
				int numNode = Integer.parseInt(element);
				numNodes[i - 1] = numNode;
			}
			return Pair.of(id, numNodes);
		};

		Map<Double, Pair<Integer, Double>> stat = new HashMap<>();
		int n = Math.min(base.size(), prop.size());
		int i = 0, j = 0;
		for (; i < n && j < n; i++, j++) {
			String b = base.get(i);
			String p = prop.get(j);

			Pair<String, int[]> br = parser.apply(b);
			Pair<String, int[]> pr = parser.apply(p);

			if (!br.getKey().equals(pr.getKey())) {
				System.err.println("Different Profile (" + n + ").");
				i--;
				j++;
				continue;
			}

			int[] bn = br.getValue();
			int[] pn = pr.getValue();

			int len = bn.length;
			if (len != pn.length) {
				System.err.println("Different Winners (" + n + ").");
				i--;
				j++;
				continue;
			}

			double imprv = 0;
			for (int k = 0; k < len; k++) {
				double winner = (k + 1.0d) / len;
				imprv = 1 - (pn[k] * 1.0d / bn[k]);
				// imprv = (bn[k] * 1.0d / pn[k]) - 1;
				// imprv = (bn[k] * 1.0d / pn[k]);

				Pair<Integer, Double> pair = null;
				if ((pair = stat.get(winner)) == null)
					pair = Pair.of(1, imprv);
				else
					pair = Pair.of(pair.getKey() + 1, pair.getValue() + imprv);
				stat.put(winner, pair);
			}
		}

		StringBuffer sb = new StringBuffer();
		List<Double> numWinner = new ArrayList<>(stat.keySet());
		Collections.sort(numWinner);

		Pair<Integer, Double> pair = null;
		for (double num : numWinner) {
			sb.append(num).append("\t");
			pair = stat.get(num);
			double imprv = pair.getValue() / pair.getKey();
			sb.append(imprv).append("\t").append(pair.getKey()).append("\r\n");
		}
		OpenOption[] options = { StandardOpenOption.CREATE, StandardOpenOption.WRITE,
				StandardOpenOption.TRUNCATE_EXISTING };
		Files.write(improved, sb.toString().getBytes(), options);
	}

	/**
	 * Quick summary on the winners' early discovery
	 * 
	 * @param input
	 * @param output
	 * @throws IOException
	 */
	public static void earlyDiscovery(Path input, Path output) throws IOException {
		int num = 100;
		String meta = "name";
		for (int i = 0; i <= num; i++) {
			meta += "," + i;
		}
		meta += "\r\n";

		Files.write(output, meta.getBytes(), options);
		Files.lines(input).forEach(line -> {
			int ind = line.indexOf(";");
			StringBuffer sb = new StringBuffer();
			String fname = line.substring(0, ind);
			sb.append(fname);
			System.out.println(fname);

			ind = line.indexOf("[", ind);
			String winners = line.substring(ind + 1, line.indexOf("]"));
			String[] fields = winners.split(",");
			int nw = fields.length;

			ind = line.lastIndexOf("[", ind);
			String pct = line.substring(ind + 1).replace("]", "");
			fields = pct.split(",");

			double d = 0.01;
			double[] data = new double[num + 1];
			data[0] = 0;
			for (int i = 0; i < fields.length; i++) {
				String field = fields[i].trim();
				double npct = Double.parseDouble(field);
				double wpct = (i + 1) / nw;
				for (int j = 0; j <= 100; j++) {
					if (npct == j * d)
						data[j] = wpct;
					else if (j * d < npct && npct < (j + 1) * d)
						data[j] = wpct;
				}
			}

			for (int j = 0; j <= num; j++)
				sb.append("," + data[j]);
			sb.append("\r\n");
			try {
				Files.write(output, sb.toString().getBytes(), options);
			} catch (IOException e) {
				e.printStackTrace();
			}
		});
	}

	public static void main1111(String[] args) throws IOException {
		String base = "/Users/chjiang/Documents/csc/";
		String dataset = "soc-3", rule = "coombs-hardcase";

		List<Path> files = Files.list(Paths.get(base + dataset + "-" + rule + "/")).collect(Collectors.toList());
		Profile<Integer> profile = null;
		for (Path file : files) {
			profile = DataEngine.loadProfile(file);
			int hardness = VoteLab.getHardness("coombs", profile);
			if (hardness == 1)
				System.out.println(file.toFile().getName());
		}
	}

	public static void main34(String[] args) throws IOException {
		TickClock.beginTick();

		String base = "/Users/chjiang/Documents/csc/";
		String dataset = "soc-4", rule = "baldwin";

		List<Path> files = Files.list(Paths.get(base + dataset)).collect(Collectors.toList());
		Path output = Paths.get(base + dataset + rule + "-hardness.txt");
		VoteLab.hardness(rule, files, output);
		TickClock.stopTick();
	}

	public static void main2(String[] args) throws IOException {
		String meta = "Usage Example - 1: java -jar stv.jar m=50:90 n=10:100 k=1:10 h=1 c=1 p=1 s=0 q=0 r=1\n"
				+ "Usage Example - 2: java -jar stv.jar m=50 n=100 k=1:10 h=1 c=0 p=1 s=1 q=1 r=1";

		int numParameter = 9;
		if (args == null || args.length != numParameter) {
			System.out.println("You are required to provide at least " + numParameter + " parameters:");
			System.out.println(meta);
			return;
		}

		int[] m = new int[2];
		String[] arg = args[0].trim().substring(2).split(":");
		m[0] = Integer.parseInt(arg[0]);
		if (arg.length == 1)
			m[1] = m[0];
		else if (arg.length == 2)
			m[1] = Integer.parseInt(arg[1]);
		else {
			System.out.println("You are required to give correct format for m:");
			System.out.println(meta);
			return;
		}

		int[] n = new int[2];
		arg = args[1].trim().substring(2).split(":");
		n[0] = Integer.parseInt(arg[0]);
		if (arg.length == 1)
			n[1] = n[0];
		else if (arg.length == 2)
			n[1] = Integer.parseInt(arg[1]);
		else {
			System.out.println("You are required to give correct format for n:");
			System.out.println(meta);
			return;
		}

		int[] k = new int[2];
		arg = args[2].trim().substring(2).split(":");
		k[0] = Integer.parseInt(arg[0]);
		if (arg.length == 1)
			k[1] = k[0];
		else if (arg.length == 2)
			k[1] = Integer.parseInt(arg[1]);
		else {
			System.out.println("You are required to give correct format for k:");
			System.out.println(meta);
			return;
		}

		int h = Integer.parseInt(args[3].trim().substring(2));
		int c = Integer.parseInt(args[4].trim().substring(2));
		int p = Integer.parseInt(args[5].trim().substring(2));
		int s = Integer.parseInt(args[6].trim().substring(2));
		int q = Integer.parseInt(args[7].trim().substring(2));
		int r = Integer.parseInt(args[8].trim().substring(2));

		String base = "./";
		String dataset = "soc-3";

		if (!Files.exists(Paths.get(base + dataset + "-hardness.txt")))
			VoteLab.getHardCases("stv", base, dataset);

		VoteResearch shr = new VoteResearch();
		boolean heuristic = (h == 1), cache = (c == 1), pruning = (p == 1), sampling = (s == 1), queue = (q == 1),
				recursive = (r == 1);
		shr.eval(base, dataset, m, n, k, heuristic, cache, pruning, sampling, queue, recursive);
	}

	public static void main1234(String[] args) throws IOException {
		TickClock.beginTick();

		int[] m = { 10, 30 }, n = { 10, 100 }, k = { 1, 1000 };

		String base = "/users/chjiang/documents/csc/";
		String dataset = "soc-3";

		boolean c = true, p = true, r = false;
		boolean h = false, s = false;

		int pf = 8;
		VoteLab.experiment("stv", base, dataset, m, n, k, h, c, p, s, r, pf);

		// m = n = k = null;
		// VoteLab.experiment("stv", base, dataset, null, null, null, h, c, p,
		// s, r, pf);

		TickClock.stopTick();
	}

	public static void main11112(String[] args) throws IOException {
		TickClock.beginTick();

		Path path = Paths.get("/Users/chjiang/Documents/csc/jun/soc-4-p0h0s0c1.txt");
		List<String> lines = Files.readAllLines(path);
		Map<Integer, List<Integer>> map = new HashMap<>();
		for (String line : lines) {
			int ind = line.indexOf(";");
			StringBuffer sb = new StringBuffer();
			String fname = line.substring(0, ind);
			sb.append(fname);

			int m = -1;
			if (fname.contains("M10"))
				m = 10;
			else if (fname.contains("M20"))
				m = 20;
			else
				m = 30;

			List<Integer> values = null;
			if ((values = map.get(m)) == null)
				values = new ArrayList<>();

			ind = line.indexOf("[", ind);
			String winners = line.substring(ind + 1, line.indexOf("]"));
			String[] fields = winners.split(",");
			for (String field : fields)
				values.add(Integer.parseInt(field.trim()));
			map.put(m, values);
		}

		Path output = Paths.get("/Users/chjiang/Documents/csc/single-peaked.csv");

		StringBuffer sb = new StringBuffer();
		sb.append("m\ti\tw\n");
		for (int m : map.keySet()) {
			int size = map.get(m).size();
			List<Integer> winners = map.get(m);
			for (int i = 0; i < size; i++) {
				sb.append(m + "\t" + i + "\t" + winners.get(i) + "\n");
			}
		}
		Files.write(output, sb.toString().getBytes(), options);

		TickClock.stopTick();
	}

	public static void main17777(String[] args) throws IOException {
		TickClock.beginTick();

		String base = "/users/chjiang/documents/csc/soc-3-";

		// int[] ind = { 2, 3, 4 };
		// for (int pf : ind) {
		// Path baseline = Paths.get(base + "stv-k1-1000-h0c1p1s0r0pf0.txt");
		// Path proposed = Paths.get(base + "stv-k1-1000-h0c1p1s0r0pf" + pf +
		// ".txt");
		// Path improved = Paths.get(base + "stv-k1-1000-h0c1p1s0r0pf0-pf" + pf
		// + ".txt");
		// VoteLab.reducedNumNode(baseline, proposed, improved);
		// }

		int pf = 9;
		Path baseline = Paths.get(base + "stv-k1-1000-h0c1p1s0r0pf0.txt");
		Path proposed = Paths.get(base + "stv-k1-1000-h0c1p1s0r0pf" + pf + ".txt");
		Path improved = Paths.get(base + "stv-k1-1000-h0c1p1s0r0pf0-pf" + pf + ".txt");
		VoteLab.reducedNumNode(baseline, proposed, improved);

		// VoteLab.predictionEval(Paths.get(base +
		// "stv-k1-1000-h0c1p1s0r0pf0.txt"));

		TickClock.stopTick();
	}

	public static void main2323(String[] args) throws IOException {
		String base = "/users/chjiang/documents/csc/", rule = "coombs";
		String dataset = "soc-3-" + rule;
		VoteLab.getHardCases(rule, base, dataset);
	}

	public static void main12(String[] args) throws IOException {
		String base = "/users/chjiang/documents/csc/", rule = "baldwin";
		String dataset = "soc-3-" + rule;

		int[] ms = { 10, 20, 30 };
		Path src, dest;
		CopyOption[] cpOptions = { StandardCopyOption.COPY_ATTRIBUTES, StandardCopyOption.REPLACE_EXISTING };
		for (int m : ms) {
			for (int n = 10; n <= 100; n += 10) {
				int count = 2000;
				for (int k = 1; k <= 3000; k++) {
					String name = "M" + m + "N" + n + "-";
					src = Paths.get(base + dataset + "-hardcase/" + name + k + ".csv");
					if (Files.exists(src)) {
						dest = Paths.get(base + dataset + "-argument/" + name + count + ".csv");
						Files.copy(src, dest, cpOptions);
						count++;
					}
				}
			}
		}
	}

	public static void main(String[] args) throws IOException {
		TickClock.beginTick();

		String base = "/users/chjiang/documents/csc/";
		// Path outcome = Paths.get(base + "/soc-5-stv-h0c1p1s0r0pf0.txt");
		// Path dataFile =
		// Paths.get("/users/chjiang/documents/csc/soc-5-stv.dat");
		// VoteLab.LearnVotingRule.getDataSet(base + "/soc-5-stv-hardcase/",
		// outcome, dataFile, 0);
		// List<String> lines =
		// Files.readAllLines(dataFile).stream().distinct().collect(Collectors.toList());
		String trainFile = base + "/soc-5-stv.csv";
		// Path trainPath = Paths.get(trainFile);
		// Files.write(trainPath, lines);

		LearnedRule rule = learn(trainFile);
		predict(base + "/soc-3-hardcase/", Paths.get(base + "soc-3-stv-k1-1000-h0c1p1s0r0pf3.txt"), rule);
		// experiment(base, "soc-3", trainFile);

		TickClock.stopTick();
	}

	public static LogisticRegression train(String trainFile) {
		SampleSet sampleset = Data.loadSampleSet(trainFile);
		int size = sampleset.size(), dim = sampleset.dim();
		double[][] input = new double[size][dim];
		int[] output = new int[size];
		for (int i = 0; i < size; i++) {
			input[i] = sampleset.getSample(i).getFeatures();
			output[i] = sampleset.getLabel(i);
		}
		LogisticRegression.Trainer trainer = new LogisticRegression.Trainer();
		LogisticRegression algo = trainer.train(input, output);
		return algo;
	}

	public static void experiment(String base, String dataset, String trainFile) throws IOException {
		boolean heuristic = false, cache = true, pruning = true, sampling = false, recursive = false;
		int pFunction = 9;
		STVPlus2 rule = new STVPlus2(heuristic, cache, pruning, sampling, recursive, pFunction);
		String hp = base + dataset + "-stv-k1-1000";

		int h = heuristic ? 1 : 0, c = cache ? 1 : 0, p = pruning ? 1 : 0, s = sampling ? 1 : 0, r = recursive ? 1 : 0;
		hp += "-h" + h + "c" + c + "p" + p + "s" + s + "r" + r + "pf" + pFunction + ".txt";
		rule.model = train(trainFile);

		hp += "-h" + h + "c" + c + "p" + p + "s" + s + "r" + r + "pf" + pFunction + ".txt";

		for (int m = 10; m <= 10; m += 10) {
			for (int n = 10; n <= 100; n += 10) {
				Path file = null;
				Profile<Integer> profile;
				int count = 0;

				Path output = Paths.get(hp);

				for (int i = 1; i <= 2000; i++) {
					String name = "M" + m + "N" + n + "-" + i + ".csv";
					file = Paths.get(base + dataset + "-hardcase/" + name);
					if (Files.exists(file)) {
						count++;
						if (count < 1)
							continue;
						else if (count > 1000)
							return;

						System.out.println(name + "\t" + ZonedDateTime.now().format(fmt));

						profile = DataEngine.loadProfile(file);
						List<Integer> winners = rule.getAllWinners(profile);
						report(rule, winners, name, sampling, output);
					}
				}
			}
		}
	}

	public static LearnedRule learn(String trainFile) {
		SampleSet sampleset = Data.loadSampleSet(trainFile);
		int size = sampleset.size(), dim = sampleset.dim();
		double[][] input = new double[size][dim];
		int[] output = new int[size];
		for (int i = 0; i < size; i++) {
			input[i] = sampleset.getSample(i).getFeatures();
			output[i] = sampleset.getLabel(i);
		}

		LogisticRegression.Trainer trainer = new LogisticRegression.Trainer();
		LogisticRegression algo = trainer.train(input, output);

		Function<Profile<Integer>, List<Integer>> mechanism = profile -> {
			List<Integer> items = Arrays.asList(profile.getSortedItems());
			double[][] features = DataEngine.getFeatures(profile, items);
			List<Integer> winners = new ArrayList<>();
			for (int i = 0; i < items.size(); i++) {
				double[] x = features[i];
				int pred = algo.predict(x);
				if (pred == 1)
					winners.add(items.get(i));
			}
			return winners;
		};
		LearnedRule learnedRule = new LearnedRule(mechanism);
		return learnedRule;
	}

	public static void predict(String base, Path outcome, LearnedRule rule) throws IOException {
		Map<String, List<Integer>> outcomes = VoteLab.parseElectionOutcome(outcome);
		Profile<Integer> profile;
		List<Integer> preds, truth;
		int correct = 0, total = 0;

		StringBuffer sb = new StringBuffer();
		int batch = 1000;
		Path dest = Paths.get("/users/chjiang/documents/csc/soc-0-stv-pred.csv");
		for (String name : outcomes.keySet()) {
			profile = DataEngine.loadProfile(Paths.get(base + name));
			preds = rule.getAllWinners(profile);

			truth = outcomes.get(name);
			for (int pred : preds)
				if (truth.contains(pred))
					correct++;

			total += truth.size();

			if (batch > 0) {
				sb.append(name + "\t" + truth + "\t" + preds + "\n");
				batch--;
			} else {
				sb = new StringBuffer();
				batch = 1000;
				Files.write(dest, sb.toString().getBytes(), options);
			}
			System.out.println(name);
		}

		if (batch > 0) {
			Files.write(dest, sb.toString().getBytes(), options);
		}
		System.out.println(correct * 1.0 / total);
	}
}
