package com.horsehour.vote;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.OpenOption;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Stream;

import com.horsehour.util.MulticoreExecutor;
import com.horsehour.util.TickClock;
import com.horsehour.vote.data.VoteLab;

/**
 *
 * @author Chunheng Jiang
 * @version 1.0
 * @since 3:25:08 PM, Dec 28, 2016
 *
 */

public class VoteDataSet {
	public static String dataset1 = "/Users/chjiang/Documents/csc/soc-1/";
	public static String dataset2 = "/Users/chjiang/Documents/csc/soc-2/";
	public static String dataset3 = "/Users/chjiang/Documents/csc/soc-3/";
	public static String dataset4 = "/Users/chjiang/Documents/csc/soc-4/";
	public static String dataset5 = "/Users/chjiang/Documents/csc/soc-5/";

	public static void generateFullProfileSpace(int numItem) throws IOException {
		DataEngine.generateFullProfileSpace(numItem, dataset1);
	}

	public static void generateFullProfileSpace(int m, int n) {
		StringBuffer meta = new StringBuffer();
		meta.append(m + "\n");
		for (int i = 0; i < m; i++)
			meta.append(i + ",c" + i + "\n");

		OpenOption[] options = { StandardOpenOption.CREATE, StandardOpenOption.APPEND, StandardOpenOption.WRITE };
		Stream<Profile<Integer>> stream = DataEngine.getPreferenceProfiles(m, n, false);
		AtomicLong count = new AtomicLong(0);
		stream.forEach(profile -> {
			count.incrementAndGet();
			StringBuffer sb = new StringBuffer();
			sb.append(meta);
			sb.append(n + "," + n + "," + profile.data.length + "\n");

			Integer[][] data = profile.data;
			for (int i = 0; i < data.length; i++) {
				sb.append(profile.votes[i]);
				for (int k = 0; k < data[i].length; k++)
					sb.append("," + data[i][k]);
				sb.append("\n");
			}
			String name = "M" + m + "N" + n + "-" + count + ".csv";
			try {
				String socFile = dataset2 + "/" + name;
				Files.write(Paths.get(socFile), sb.toString().getBytes(), options);
			} catch (IOException e) {
				e.printStackTrace();
				return;
			}
		});
	}

	/**
	 * Generate random profiles
	 * 
	 * @param nc
	 *            number of candidates
	 * @param nv
	 *            number of votes
	 * @param ns
	 *            number of samples
	 * @throws Exception
	 */
	public static void generateRandomProfiles(List<Integer> nc, List<Integer> nv, int ns) throws Exception {
		int numThread = 5;
		List<RandomProfileGTask> tasks = null;
		for (int m : nc) {
			tasks = new ArrayList<>();
			for (int i = 0; i < numThread; i++)
				tasks.add(new RandomProfileGTask(m, ns));

			for (int n : nv) {
				tasks.get(n % numThread).nv.add(n);
			}
			MulticoreExecutor.run(tasks);
		}
	}

	public static class RandomProfileGTask implements Callable<Void> {
		public List<Integer> nv = null;
		public int m = -1, s = 1;

		public RandomProfileGTask(int m, int s) {
			this.m = m;
			this.s = s;
			nv = new ArrayList<>();
		}

		@Override
		public Void call() throws Exception {
			for (int n : nv) {
				DataEngine.generateRandomProfiles(m, n, s, dataset3);
				System.out.printf("m=%d,n=%d\n", m, n);
			}
			return null;
		}
	}

	public static void main(String[] args) throws Exception {
		TickClock.beginTick();

		// List<Integer> nc = Arrays.asList(10, 20, 30);
		// List<Integer> nv = new ArrayList<>();
		//
		// for (int i = 1; i <= 10; i++) {
		// nv.add(10 * i);
		// }
		// VoteDataSet.generateRandomProfiles(nc, nv, 1000);

		// String baseFile = "/users/chjiang/documents/csc/soc-5-stv/";
		// List<Integer> nList = MathLib.Rand.sample(1, 101, 30);
		// for (int m = 3; m <= 30; m++) {
		// for (int n : nList)
		// DataEngine.getHardCases("stv", m, n, 100, baseFile);
		// }

		String base = "/users/chjiang/documents/csc/";
		String dataset = "soc-5-stv";
		int[] ms = {3, 30};
		int[] ns = {1, 100};

		boolean heuristic = false, cache = true, pruning = true;
		boolean sampling = false, recursive = false;

		VoteLab.experiment("stv", base, dataset, ms, ns, null, heuristic, cache, pruning, sampling, recursive, 0);

		TickClock.stopTick();
	}
}