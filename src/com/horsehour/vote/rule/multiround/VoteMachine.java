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
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import com.horsehour.util.MathLib;
import com.horsehour.util.TickClock;
import com.horsehour.vote.Profile;
import com.horsehour.vote.data.DataEngine;

/**
 *
 * Vote machine defines the basic operation to learn a vote rule with machine
 * learning approaches
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 2:32:46 PM, Feb 1, 2017
 *
 */

public class VoteMachine {
	String base = "/Users/chjiang/Documents/csc/soc-2/";
	OpenOption[] options = { StandardOpenOption.APPEND, StandardOpenOption.CREATE, StandardOpenOption.WRITE };

	/**
	 * Create data set from known election records
	 * 
	 * @param electionRecords
	 *            election outcomes, including the profile name, the winners and
	 *            other data
	 * @param dataFile
	 *            destination to keep all the data set.
	 * @param nSample
	 *            number of profiles used to create data set. If nSample <= 0,
	 *            all profiles will be used to create the data set
	 * @throws IOException
	 */
	void getDataSet(Path electionRecords, Path dataFile, int nSample) {
		List<String> lines = null;
		try {
			lines = Files.readAllLines(electionRecords);
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

		Set<String> distinct = new HashSet<>();
		for (int i : index) {
			String line = lines.get(i);
			int idx = line.indexOf("\t");
			String name = line.substring(0, idx);
			profile = DataEngine.loadProfile(Paths.get(base + name));
			double[][] features = DataEngine.getFeatures(profile, Arrays.asList(profile.getSortedItems()));

			List<Integer> winners = new ArrayList<>();
			for (String field : line.split("\t")) {
				if (field.startsWith("[") && field.endsWith("]")) {
					field = field.replace("[", "").replace("]", "");

					// unique winner
					if (!field.contains(",")) {
						winners.add(Integer.parseInt(field));
						break;
					}

					// more than 2 winners
					for (String winner : field.split(","))
						winners.add(Integer.parseInt(winner.trim()));
					break;
				}
			}

			for (int k = 0; k < features.length; k++) {
				String content = "";
				for (double feature : features[k])
					content += feature + ",";

				if (winners.contains(k))
					content += "1";
				else
					content += "0";
				distinct.add(content);
			}
		}

		StringBuffer sb = new StringBuffer();
		for (String content : distinct)
			sb.append(content).append("\r\n");

		try {
			Files.write(dataFile, sb.toString().getBytes(), options);
		} catch (IOException e) {
			e.printStackTrace();
			return;
		}
	}

	void recordTrace(boolean heuristic, boolean cache, boolean pruning, boolean sampling, boolean recursive,
			int pFunction) throws IOException {
		String base = "/Users/chjiang/Documents/csc/";
		String dataset = "soc-4";
		STVPlus2 rule = new STVPlus2(heuristic, cache, pruning, sampling, recursive, pFunction);
		String hp = base + dataset + "-k1-1000";
		int h = heuristic ? 1 : 0, c = cache ? 1 : 0, p = pruning ? 1 : 0, s = sampling ? 1 : 0, r = recursive ? 1 : 0;
		hp += "-h" + h + "c" + c + "p" + p + "s" + s + "r" + r + "pf" + pFunction + "-trace.txt";

		Path output = Paths.get(hp);

		DateTimeFormatter fmt = DateTimeFormatter.ofPattern("yyyyMMdd HH:mm:ss.SSSSSS");
		OpenOption[] options = { StandardOpenOption.APPEND, StandardOpenOption.CREATE, StandardOpenOption.WRITE };

		Path file;
		Profile<Integer> profile;

		StringBuffer sb = new StringBuffer();

		for (int m = 10; m <= 50; m += 10) {
			for (int n = 10; n <= 100; n += 10) {
				int count = 0;
				for (int k = 1; k <= 1500; k++) {
					String name = "M" + m + "N" + n + "-" + k + ".csv";
					// file = Paths.get(base + dataset + "-hardcase/" + name);
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

	void extractTrace(Path input, Path output) throws IOException {
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

	public static void main(String[] args) throws IOException {
		TickClock.beginTick();

		String base = "/Users/chjiang/Documents/csc/";
		String[] names = { "STV-soc-3-k1-1000-h0c1p1s1r0pf0-trace" };

		VoteMachine vm = new VoteMachine();
		for (String name : names) {
			Path input = Paths.get(base + name + ".txt");
			Path output = Paths.get(base + name + "-format.txt");
			vm.extractTrace(input, output);
		}

		TickClock.stopTick();
	}
}
