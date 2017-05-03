package com.horsehour.vote.rule.multiround;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.OpenOption;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import org.apache.commons.lang3.tuple.Pair;

import com.horsehour.util.MathLib;
import com.horsehour.util.TickClock;
import com.horsehour.vote.Profile;
import com.horsehour.vote.data.DataEngine;
import com.horsehour.vote.data.VoteLab;

/**
 * Research voting problems
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 9:13:57 PM, Jan 4, 2017
 */
public class VoteResearch {
	OpenOption[] options = { StandardOpenOption.APPEND, StandardOpenOption.CREATE, StandardOpenOption.WRITE };
	DateTimeFormatter fmt = DateTimeFormatter.ofPattern("yyyyMMdd HH:mm:ss.SSSSSS");

	public VoteResearch() {}

	public void eval(String base, String dataset, int[] ms, int[] ns, int[] k, boolean heuristic, boolean cache,
			boolean pruning, boolean sampling, boolean recursive) throws IOException {
		for (int m = ms[0]; m <= ms[1]; m++)
			for (int n = ns[0]; n <= ns[1]; n++)
				eval(base, dataset, m, n, k, heuristic, cache, pruning, sampling, recursive);
	}

	public void eval(String base, String dataset, int m, int n, int[] k, boolean heuristic, boolean cache,
			boolean pruning, boolean sampling, boolean recursive) throws IOException {
		StringBuffer sb = new StringBuffer();
		String hp = base + dataset + "-k" + k[0] + "-" + k[1];
		int h = heuristic ? 1 : 0, c = cache ? 1 : 0, p = pruning ? 1 : 0, s = sampling ? 1 : 0, r = recursive ? 1 : 0;
		hp += "-h" + h + "c" + c + "p" + p + "s" + s + "r" + r + ".txt";

		Path output = Paths.get(hp);

		int count = 0;
		Path file = null;
		Profile<Integer> profile;

		STV rule = new STV(heuristic, cache, pruning, sampling, recursive);
		for (int i = 1; i <= 2000; i++) {
			String name = "M" + m + "N" + n + "-" + i + ".csv";
			file = Paths.get(base + dataset + "-hardcase/" + name);
			if (Files.exists(file)) {
				count++;
				if (count < k[0])
					continue;
				else if (count > k[1])
					return;

				System.out.println(name + "\t" + ZonedDateTime.now().format(fmt));

				profile = DataEngine.loadProfile(file);
				List<Integer> winners = rule.getAllWinners(profile);

				String format = "%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%s\t%s";
				float ratio = rule.timeScoring + rule.timeCacheEval + rule.timeHeuristicEval + rule.timeFork
						+ rule.timeSelectNext + rule.timePruneEval;

				if (rule.time == 0)
					ratio = 1.0F;
				else
					ratio /= rule.time;

				if (sampling) {
					sb.append(String.format(format + "\t%s\n", name, rule.numItemTotal, rule.numFailH, rule.numSingleL,
							rule.numMultiL, rule.numScoring, rule.numNodeWH, rule.numNodeWOH, rule.numNode,
							rule.numNodeFull, rule.numCacheHit, rule.numCacheMiss, rule.numPruneHit, rule.numPruneMiss,
							rule.time, rule.timeScoring, rule.timeHeuristicEval, rule.timeCacheEval, rule.timePruneEval,
							rule.timeSampling, rule.timeSelectNext, rule.timeFork, ratio, winners, rule.freq,
							rule.electedSampling));
				} else {
					sb.append(String.format(format + "\n", name, rule.numItemTotal, rule.numFailH, rule.numSingleL,
							rule.numMultiL, rule.numScoring, rule.numNodeWH, rule.numNodeWOH, rule.numNode,
							rule.numNodeFull, rule.numCacheHit, rule.numCacheMiss, rule.numPruneHit, rule.numPruneMiss,
							rule.time, rule.timeScoring, rule.timeHeuristicEval, rule.timeCacheEval, rule.timePruneEval,
							rule.timeSampling, rule.timeSelectNext, rule.timeFork, ratio, winners, rule.freq));
				}
				Files.write(output, sb.toString().getBytes(), options);
				sb = new StringBuffer();
			}
		}
	}

	public void eval(String base, String dataset, int[] ms, int[] ns, int[] k, boolean heuristic, boolean cache,
			boolean pruning, boolean sampling, boolean queue, boolean recursive) throws IOException {
		for (int m = ms[0]; m <= ms[1]; m++)
			for (int n = ns[0]; n <= ns[1]; n++)
				eval(base, dataset, m, n, k, heuristic, cache, pruning, sampling, queue, recursive);
	}

	public void eval(String base, String dataset, int m, int n, int[] k, boolean heuristic, boolean cache,
			boolean pruning, boolean sampling, boolean queue, boolean recursive) throws IOException {
		StringBuffer sb = new StringBuffer();
		String hp = base + dataset + "-k" + k[0] + "-" + k[1];
		int h = heuristic ? 1 : 0, c = cache ? 1 : 0, p = pruning ? 1 : 0, s = sampling ? 1 : 0, q = queue ? 1 : 0,
				r = recursive ? 1 : 0;
		hp += "-h" + h + "c" + c + "p" + p + "s" + s + "q" + q + "r" + r + ".txt";

		Path output = Paths.get(hp);

		int count = 0;
		Path file = null;
		Profile<Integer> profile;

		STVPlus rule = new STVPlus(heuristic, cache, pruning, sampling, queue, recursive);

		for (int i = 1; i <= 2000; i++) {
			String name = "M" + m + "N" + n + "-" + i + ".csv";
			file = Paths.get(base + dataset + "-hardcase/" + name);
			if (Files.exists(file)) {
				count++;
				if (count < k[0])
					continue;
				else if (count > k[1])
					return;

				System.out.println(name + "\t" + ZonedDateTime.now().format(fmt));

				profile = DataEngine.loadProfile(file);
				List<Integer> winners = rule.getAllWinners(profile);

				String format = "%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%s\t%s";
				float ratio = rule.timeScoring + rule.timeCacheEval + rule.timeHeuristicEval + rule.timeFork
						+ rule.timeSelectNext + rule.timePruneEval + rule.timePred + rule.timeComputePriority;

				if (rule.time == 0)
					ratio = 1.0F;
				else
					ratio /= rule.time;

				if (sampling) {
					sb.append(String.format(format + "\t%s\n", name, rule.numItemTotal, rule.numFailH, rule.numSingleL,
							rule.numMultiL, rule.numScoring, rule.numNodeWH, rule.numNodeWOH, rule.numNode,
							rule.numNodeFull, rule.numCacheHit, rule.numCacheMiss, rule.numPruneHit, rule.numPruneMiss,
							rule.time, rule.timeScoring, rule.timeHeuristicEval, rule.timeCacheEval, rule.timePruneEval,
							rule.timeSampling, rule.timeSelectNext, rule.timeFork, rule.timePred,
							rule.timeComputePriority, ratio, winners, rule.freq, rule.electedSampling));
				} else {
					sb.append(String.format(format + "\n", name, rule.numItemTotal, rule.numFailH, rule.numSingleL,
							rule.numMultiL, rule.numScoring, rule.numNodeWH, rule.numNodeWOH, rule.numNode,
							rule.numNodeFull, rule.numCacheHit, rule.numCacheMiss, rule.numPruneHit, rule.numPruneMiss,
							rule.time, rule.timeScoring, rule.timeHeuristicEval, rule.timeCacheEval, rule.timePruneEval,
							rule.timeSampling, rule.timeSelectNext, rule.timeFork, rule.timePred,
							rule.timeComputePriority, ratio, winners, rule.freq));
				}
				Files.write(output, sb.toString().getBytes(), options);
				sb = new StringBuffer();
			}
		}
	}

	/**
	 * @param input
	 * @param heuristic
	 * @param cache
	 * @param pruning
	 * @param sampling
	 * @param recursive
	 * @return Evaluation the election outcomes with STV, w/o the heurisitic
	 *         searching algorithm, w/o the pruning approach, w/o the priority
	 *         sampling technique
	 */
	public String eval(Path input, boolean heuristic, boolean cache, boolean pruning, boolean sampling,
			boolean recursive) {
		STV rule = new STV(heuristic, cache, pruning, sampling, recursive);

		Profile<Integer> profile = DataEngine.loadProfile(input);
		List<Integer> winners = rule.getAllWinners(profile);

		StringBuffer sb = new StringBuffer();
		String name = input.toFile().getName();

		String format = "%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%s\t%s";
		float ratio = rule.timeScoring + rule.timeCacheEval + rule.timeHeuristicEval + rule.timeFork
				+ rule.timeSelectNext + rule.timePruneEval;

		if (rule.time == 0)
			ratio = 1.0F;
		else
			ratio /= rule.time;

		if (sampling) {
			sb.append(String.format(format + "\t%s", name, rule.numItemTotal, rule.numFailH, rule.numSingleL,
					rule.numMultiL, rule.numScoring, rule.numNodeWH, rule.numNodeWOH, rule.numNode, rule.numNodeFull,
					rule.numCacheHit, rule.numCacheMiss, rule.numPruneHit, rule.numPruneMiss, rule.time,
					rule.timeScoring, rule.timeHeuristicEval, rule.timeCacheEval, rule.timePruneEval, rule.timeSampling,
					rule.timeSelectNext, rule.timeFork, ratio, winners, rule.freq, rule.electedSampling));
		} else {
			sb.append(String.format(format, name, rule.numItemTotal, rule.numFailH, rule.numSingleL, rule.numMultiL,
					rule.numScoring, rule.numNodeWH, rule.numNodeWOH, rule.numNode, rule.numNodeFull, rule.numCacheHit,
					rule.numCacheMiss, rule.numPruneHit, rule.numPruneMiss, rule.time, rule.timeScoring,
					rule.timeHeuristicEval, rule.timeCacheEval, rule.timePruneEval, rule.timeSampling,
					rule.timeSelectNext, rule.timeFork, ratio, winners, rule.freq));
		}
		return sb.toString();
	}

	public void eval(String base, String dataset, boolean heuristic, boolean cache, boolean pruning, boolean sampling,
			boolean recursive) throws IOException {
		String hp = base + dataset;
		int h = heuristic ? 1 : 0, c = cache ? 1 : 0, p = pruning ? 1 : 0, s = sampling ? 1 : 0, r = recursive ? 1 : 0;
		hp += "-h" + h + "c" + c + "p" + p + "s" + s + "r" + r + ".txt";

		Path output = Paths.get(hp);
		STV rule = new STV(heuristic, cache, pruning, sampling, recursive);

		Path input = Paths.get(base + "/" + dataset);
		Files.list(input).forEach(path -> {
			String name = path.toFile().getName();
			Profile<Integer> profile = DataEngine.loadProfile(path);
			StringBuffer sb = new StringBuffer();
			sb.append(name + "\t" + rule.getAllWinners(profile) + "\n");

			System.out.println(name + "\t" + ZonedDateTime.now().format(fmt));

			try {
				Files.write(output, sb.toString().getBytes(), options);
			} catch (IOException e) {
				e.printStackTrace();
			}
		});
	}

	/**
	 * @param input
	 * @param output
	 * @throws IOException
	 */
	public void report(Path input, Path output) throws IOException {
		List<Entry> entries = new ArrayList<>();
		for (String line : Files.readAllLines(input))
			entries.add(new Entry(line));

		int nField = entries.get(0).fieldList.size();
		Map<Integer, List<Entry>> clusters = entries.stream().collect(Collectors.groupingBy(e -> e.m));

		StringBuffer sb = new StringBuffer();
		List<Entry> list;
		Map<Integer, List<Entry>> children;

		List<Double> init = new ArrayList<>();
		for (int i = 0; i < nField; i++)
			init.add(0.0);

		List<Integer> mList = new ArrayList<>(clusters.keySet());
		List<Integer> nList = null;
		Collections.sort(mList);
		for (int m : mList) {
			list = clusters.get(m);
			children = list.stream().collect(Collectors.groupingBy(e -> e.n));

			nList = new ArrayList<>(children.keySet());
			Collections.sort(nList);

			for (int n : nList) {
				List<Double> avgn = new ArrayList<>(init);
				for (Entry entry : children.get(n))
					avgn = MathLib.Matrix.add(avgn, entry.fieldList);
				int size = children.get(n).size();
				avgn = MathLib.Matrix.multiply(avgn, 1.0d / size);

				sb.append(String.format("%d\t%d", m, n));
				for (double avg : avgn)
					sb.append("\t" + avg);
				sb.append("\n");
			}
		}

		if (Files.exists(output))
			Files.delete(output);

		String header = "m\tn\tnumFailH\tnumSingleL\tnumMultiL\tnumScoring\t"
				+ "numNodeWH\tnumNodeWOH\tnumNode\tnumNodeFull\tnumCacheHit\t"
				+ "numCacheMiss\tnumPruneHit\tnumPruneMiss\ttime\ttimeScoring\t"
				+ "timeHeuristicEval\ttimeCacheEval\ttimePruneEval\ttimeSampling\t"
				+ "timeSelectNext\ttimeGenerateNewState\tratio\n";

		String result = header + sb.toString();
		Files.write(output, result.getBytes(), options);
	}

	/**
	 * Data entry to produce report
	 */
	public class Entry {
		public int m, n, k;
		public List<Double> fieldList;

		public Entry(String line) {
			fieldList = new ArrayList<>();
			for (String cell : line.split("\t")) {
				if (cell.startsWith("M")) {
					Pattern pattern = Pattern.compile("\\d+(\\.\\d+)?");
					Matcher matcher = pattern.matcher(cell);
					int c = 0;
					while (matcher.find()) {
						int v = Integer.parseInt(matcher.group());
						if (c == 0)
							m = v;
						else if (c == 1)
							n = v;
						else
							k = v;
						c++;
					}
					continue;
				}

				if (cell.startsWith("{") || cell.startsWith("["))
					continue;

				double value = Double.parseDouble(cell);
				fieldList.add(value);
			}
		}
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
			VoteLab.getHardCases("", base, dataset);

		VoteResearch shr = new VoteResearch();
		boolean heuristic = (h == 1), cache = (c == 1), pruning = (p == 1), sampling = (s == 1), queue = (q == 1),
				recursive = (r == 1);
		shr.eval(base, dataset, m, n, k, heuristic, cache, pruning, sampling, queue, recursive);
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

	public smile.classification.LogisticRegression learnSTV() {
		String base = "/Users/chjiang/Documents/csc/";
		Path trainFile = Paths.get(base + "soc-2-h1c1p1s0r1.txt");

		List<Pair<double[], Integer>> trainset = null;
		try {
			trainset = buildTrainSet(trainFile, base);
		} catch (IOException e) {
			e.printStackTrace();
		}

		int size = trainset.size();
		int dim = trainset.get(0).getKey().length;

		Pair<double[], Integer> train;
		double[][] input = new double[size][dim];
		int[] output = new int[size];

		for (int i = 0; i < size; i++) {
			train = trainset.get(i);
			input[i] = train.getKey();
			output[i] = train.getValue();
		}

		smile.classification.LogisticRegression.Trainer trainer = null;
		trainer = new smile.classification.LogisticRegression.Trainer();
		return trainer.train(input, output);
	}

	/**
	 * Create training set from known election outcomes
	 * 
	 * @throws IOException
	 */
	List<Pair<double[], Integer>> buildTrainSet(Path trainFile, String base) throws IOException {
		Map<String, List<Integer>> election = extractElectResult(trainFile);
		Path profileFile;
		Profile<Integer> profile;

		List<Pair<double[], Integer>> trainset = new ArrayList<>();
		for (String name : election.keySet()) {
			profileFile = Paths.get(base + "/soc-2/" + name);
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
	 * @param electionFile
	 * @return map with profile file's name as the key and the corresponding
	 *         winners are the values
	 * @throws IOException
	 */
	Map<String, List<Integer>> extractElectResult(Path electionFile) throws IOException {
		Map<String, List<Integer>> map = new HashMap<>();
		List<String> lines = Files.readAllLines(electionFile);
		int n = 200000;
		for (int i : MathLib.Rand.sample(0, lines.size(), n)) {
			String line = lines.get(i);
			int idx = line.indexOf("\t");
			String name = line.substring(0, idx);

			List<Integer> winners = new ArrayList<>();
			for (String field : line.split("\t")) {
				if (field.startsWith("[") && field.endsWith("]")) {
					field = field.replace("[", "").replace("]", "");
					if (!field.contains(",")) {
						winners.add(Integer.parseInt(field));
						break;
					}
					for (String winner : field.split(","))
						winners.add(Integer.parseInt(winner.trim()));
					break;
				}
			}
			map.put(name, winners);
		}
		return map;
	}

	public static void main0(String[] args) throws IOException {
		TickClock.beginTick();

		int[] m = { 10, 20 }, n = { 10, 100 }, k = { 1, 1000 };

		String base = "/users/chjiang/documents/csc/";
		String dataset = "soc-3";

		VoteResearch shr = new VoteResearch();
		boolean heuristic = true, cache = true, pruning = true, sampling = false, training = true, recursive = false;
		shr.eval(base, dataset, m, n, k, heuristic, cache, pruning, sampling, training, recursive);

		TickClock.stopTick();
	}

	public static void main3(String[] args) throws IOException {
		TickClock.beginTick();

		String base = "/Users/chjiang/Documents/csc/";
		String dataset = "soc-3";

		boolean heuristic = false, cache = true, pruning = true, sampling = false, recursive = false;

		Path input = Paths.get(base + dataset + "/M30N30-237.csv");
		STV rule = new STV(heuristic, cache, pruning, sampling, recursive);

		String format = "#node=%d, #score=%d, t=%f, t_score=%f, t_hash=%f, t_heur=%f, t_new=%f, t_next=%f, t_eva_elected=%f, t_sample=%f, winners=%s\n";

		Profile<Integer> profile = DataEngine.loadProfile(input);
		List<Integer> winners = rule.getAllWinners(profile);

		System.out.printf(format, rule.numNode, rule.numScoring, rule.time, rule.timeScoring, rule.timeCacheEval,
				rule.timeHeuristicEval, rule.timeFork, rule.timeSelectNext, rule.timePruneEval, rule.timeSampling,
				winners);
		TickClock.stopTick();
	}

	public static void main123(String[] args) throws Exception {
		TickClock.beginTick();

		String base = "/Users/chjiang/Documents/csc/";
		String dataset = "soc-3";

		VoteResearch shr = new VoteResearch();
		List<Path> paths = new ArrayList<>();
		paths.add(Paths.get(base + dataset + "-k1-1000-h0c1p1s0r-0.txt"));
		paths.add(Paths.get(base + dataset + "-k1-1000-h1c1p1s0r-0.txt"));
		shr.consistent(paths);

		TickClock.stopTick();
	}

	public static void main2313(String[] args) throws IOException {
		TickClock.beginTick();

		String base = "/Users/chjiang/Documents/csc/";
		String dataset = "soc-2";

		boolean heuristic = true, cache = true, pruning = true, sampling = false, recursive = true;

		VoteResearch shr = new VoteResearch();
		shr.eval(base, dataset, heuristic, cache, pruning, sampling, recursive);

		TickClock.stopTick();
	}
}