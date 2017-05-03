package com.horsehour.vote.train;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.function.Function;
import java.util.stream.Collectors;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.tuple.Pair;

import com.horsehour.ml.classifier.tree.mtree.MaximumCutPlane;
import com.horsehour.util.MathLib;
import com.horsehour.util.MulticoreExecutor;
import com.horsehour.util.TickClock;
import com.horsehour.vote.ChoiceTriple;
import com.horsehour.vote.Profile;
import com.horsehour.vote.data.DataEngine;
import com.horsehour.vote.rule.Baldwin;
import com.horsehour.vote.rule.Black;
import com.horsehour.vote.rule.Borda;
import com.horsehour.vote.rule.Bucklin;
import com.horsehour.vote.rule.Condorcet;
import com.horsehour.vote.rule.Coombs;
import com.horsehour.vote.rule.Copeland;
import com.horsehour.vote.rule.InstantRunoff;
import com.horsehour.vote.rule.KemenyYoung;
import com.horsehour.vote.rule.LearnedRule;
import com.horsehour.vote.rule.Llull;
import com.horsehour.vote.rule.Maximin;
import com.horsehour.vote.rule.Nanson;
import com.horsehour.vote.rule.OklahomaVoting;
import com.horsehour.vote.rule.PairMargin;
import com.horsehour.vote.rule.Plurality;
import com.horsehour.vote.rule.RankedPairs;
import com.horsehour.vote.rule.Schulze;
import com.horsehour.vote.rule.Veto;
import com.horsehour.vote.rule.VotingRule;

/**
 *
 * @author Chunheng Jiang
 * @version 1.0
 * @since 7:17:56 PM, Nov 8, 2016
 *
 */
public class Main {
	public static VotingRule getMaximumCutPlaneRule(List<ChoiceTriple<Integer>> profiles, VotingRule oracle,
			int numItem, int numVote, int numSample, int depth) {
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
		sb.append(String.format("max_depth=%d, n_train=%d\n", depth, numSample));
		sb.append(algo.toString()).append("\n");

		String dest = String.format("csc/mcp(r=%s, m=%d, n=%d).txt", oracle.toString(), numItem, numVote);
		try {
			FileUtils.write(new File(dest), sb.toString(), "UTF8", true);
		} catch (IOException e) {
			e.printStackTrace();
		}
		return learnedRule;
	}

	public static Void learnAndEvalRule(int numItem, int numVote, int numSample, VotingRule oracle, int[] evalNumVotes)
			throws IOException {
		List<ChoiceTriple<Integer>> trainset = DataEngine.getRandomLabeledProfiles(numItem, numVote, numSample, oracle);

		int maxDepth = 6;

		long start = System.currentTimeMillis();
		VotingRule learnedRule = getMaximumCutPlaneRule(trainset, oracle, numItem, numVote, numSample, maxDepth);
		long end = System.currentTimeMillis();
		long elapsed = end - start;

		double correct = 0;
		for (ChoiceTriple<Integer> triple : trainset) {
			int winner = learnedRule.getAllWinners(triple.getProfile()).get(0);
			if (winner == triple.getWinner())
				correct += triple.size();
		}

		StringBuffer sb = new StringBuffer();
		sb.append(String.format("accu_train: %.2f (n=%d, n_profile=%d, n_distinct=%d, t=%d)\n", correct / numSample,
				numVote, numSample, trainset.size(), elapsed));

		sb.append("nv");
		for (int nv : evalNumVotes)
			sb.append(String.format(", %d", nv));
		sb.append("\n");
		sb.append(oracle.toString());

		Eval eval = new Eval();
		for (int nv : evalNumVotes) {
			double prec = eval.getSimilarity(numItem, nv, oracle, learnedRule);
			sb.append(String.format(", %.2f", prec));
		}
		sb.append("\n");

		String dest = String.format("csc/perf-mcp(r=%s, m=%d).txt", oracle.toString(), numItem, numVote);
		FileUtils.writeStringToFile(new File(dest), sb.toString(), "UTF8", true);
		return null;
	}

	public static Void learnAndEvalRule(int numItem, int numVote, int numSample, VotingRule oracle,
			List<Integer> evalNumVotes) throws IOException {
		Eval eval = new Eval();

		List<List<ChoiceTriple<Integer>>> samples = null;
		samples = eval.getSamples(numItem, numVote, Arrays.asList(numSample), oracle, null);

		List<ChoiceTriple<Integer>> trainset = null;
		trainset = samples.stream().flatMap(list -> list.stream()).collect(Collectors.toList());

		Learner learner = new Learner();
		int maxDepth = 6;
		VotingRule learnedRule = learner.getMaximumCutPlaneRule(trainset, oracle, maxDepth);
		File destFile = new File("csc/MaximumCutPlane-" + oracle.toString() + ".txt");

		// VotingRule learnedRule = learner.getLinearMachineRule(trainset,
		// oracle);
		// File destFile = new File("csc/LinearMachine-" + oracle.toString() +
		// ".txt");

		double correct = 0;
		for (ChoiceTriple<Integer> triple : trainset) {
			int winner = learnedRule.getAllWinners(triple.getProfile()).get(0);
			if (winner == triple.getWinner())
				correct += triple.size();
		}

		StringBuffer sb = new StringBuffer();
		sb.append("nv");
		for (int nv : evalNumVotes)
			sb.append(String.format(", %d", nv));
		sb.append("\n");
		sb.append(String.format("Training accuracy over %d profiles by %d voters is %.2f.\n", numSample, numVote,
				correct / numSample));

		sb.append(oracle.toString());
		for (int nv : evalNumVotes) {
			double prec = eval.getSimilarity(numItem, nv, oracle, learnedRule);
			sb.append(String.format(", %.2f", prec));
		}
		sb.append("\n");
		FileUtils.writeStringToFile(destFile, sb.toString(), "UTF8", true);
		return null;
	}

	public static void main__(String[] args) throws Exception {
		TickClock.beginTick();

		List<VotingRule> rules = new ArrayList<>();
		rules.add(new Baldwin());
		rules.add(new Black());
		rules.add(new Borda());
		rules.add(new Bucklin());
		rules.add(new Condorcet());
		rules.add(new Coombs());
		rules.add(new Copeland());
		rules.add(new InstantRunoff());
		rules.add(new KemenyYoung());
		rules.add(new Llull());
		rules.add(new Maximin());
		rules.add(new Nanson());
		rules.add(new OklahomaVoting());
		rules.add(new PairMargin());
		rules.add(new Plurality());
		rules.add(new RankedPairs());
		rules.add(new Schulze());
		rules.add(new Veto());

		int[] numSamples = MathLib.Series.range(100, 100, 10);
		int[] numVotes = MathLib.Series.range(5, 2, 15);
		int numItem = 3;

		for (int numVote : numVotes) {
			for (int numSample : numSamples) {
				if (numVote == 5 && numSample > 200)
					break;
				else if (numVote == 7 && numSample > 700)
					break;

				List<Callable<Void>> learnTaskList = new ArrayList<>();
				for (int i = 0; i < rules.size(); i++) {
					VotingRule oracle = rules.get(i);
					learnTaskList.add(() -> learnAndEvalRule(numItem, numVote, numSample, oracle, numVotes));
				}
				MulticoreExecutor.run(learnTaskList);
			}
		}

		TickClock.stopTick();
	}

	public static void main_(String[] args) throws Exception {
		TickClock.beginTick();

		List<VotingRule> rules = new ArrayList<>();
		rules.add(new Borda());
		rules.add(new Condorcet());
		rules.add(new Copeland());
		// rules.add(new InstantRunoff());
		// rules.add(new KemenyYoung());
		rules.add(new Maximin());
		rules.add(new PairMargin());
		rules.add(new Plurality());
		// rules.add(new RankedPairs());
		// rules.add(new Schulze());
		rules.add(new Veto());

		int numItem = 3, numVote = 7, maxNumVote = 31, numSample = 100;
		List<Integer> numVotes = new ArrayList<>();
		for (int nv = 5; nv <= maxNumVote; nv += 2)
			numVotes.add(nv);

		List<Callable<Void>> learnTaskList = new ArrayList<>();
		for (int i = 0; i < rules.size(); i++) {
			VotingRule oracle = rules.get(i);
			learnTaskList.add(() -> learnAndEvalRule(numItem, numVote, numSample, oracle, numVotes));
		}
		MulticoreExecutor.run(learnTaskList);

		TickClock.stopTick();
	}

	public static void main(String[] args) throws Exception {
		TickClock.beginTick();

		List<VotingRule> rules = new ArrayList<>();

		// rules.add(new Baldwin());
		// rules.add(new Black());
		// rules.add(new Borda());
		// rules.add(new Bucklin());
		rules.add(new Condorcet());
		// rules.add(new Coombs());
		// rules.add(new Copeland());
		// rules.add(new InstantRunoff());
		// rules.add(new KemenyYoung());
		// rules.add(new Llull());
		// rules.add(new Maximin());
		// rules.add(new Nanson());
		// rules.add(new OklahomaVoting());
		// rules.add(new PairMargin());
		// rules.add(new Plurality());
		// rules.add(new RankedPairs());
		// rules.add(new Schulze());
		// rules.add(new Veto());

		int numItem = 3, numVote = 5, numSample = 300;
		int[] numVotes = MathLib.Series.range(5, 2, 15);

		double c = 10;
		System.out.print("rule");
		for (int nv : numVotes)
			System.out.printf("\t" + nv);
		System.out.println();

		List<Callable<double[]>> learnTaskList = new ArrayList<>();
		for (int i = 0; i < rules.size(); i++) {
			VotingRule oracle = rules.get(i);
			List<ChoiceTriple<Integer>> profiles = DataEngine.getRandomLabeledProfiles(numItem, numVote, numSample,
					oracle);
			learnTaskList.add(() -> new Learner().getMultiClassSVMRule(profiles, c, numItem, numVotes, oracle));
		}
		List<double[]> results = MulticoreExecutor.run(learnTaskList);

		for (int i = 0; i < rules.size(); i++) {
			VotingRule oracle = rules.get(i);
			System.out.print(oracle.toString());
			double[] accu = results.get(i);
			for (int v = 0; v < accu.length; v++)
				System.out.print("\t" + accu[v]);
			System.out.println();
		}

		TickClock.stopTick();
	}
}