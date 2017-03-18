package com.horsehour.vote.rule;

import java.nio.file.Files;
import java.nio.file.OpenOption;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.stream.Collectors;

import com.horsehour.util.MathLib;
import com.horsehour.util.MulticoreExecutor;
import com.horsehour.util.TickClock;
import com.horsehour.vote.DataEngine;
import com.horsehour.vote.Profile;
import com.horsehour.vote.ScoredItems;
import com.horsehour.vote.axiom.CondorcetCriterion;
import com.horsehour.vote.axiom.CondorcetLoserCriterion;
import com.horsehour.vote.axiom.ConsistencyCriterion;
import com.horsehour.vote.axiom.HomogeneityCriterion;
import com.horsehour.vote.axiom.MajorityCriterion;
import com.horsehour.vote.axiom.MajorityLoserCriterion;
import com.horsehour.vote.axiom.MonotonicityCriterion;
import com.horsehour.vote.axiom.NeutralityCriterion;
import com.horsehour.vote.axiom.ParticipationCriterion;
import com.horsehour.vote.axiom.PluralityCriterion;
import com.horsehour.vote.axiom.ReversalSymmetryCriterion;
import com.horsehour.vote.axiom.SchwarzCriterion;
import com.horsehour.vote.axiom.SmithCriterion;
import com.horsehour.vote.axiom.VotingAxiom;
import com.horsehour.vote.train.Eval1;

/**
 * Voting Rule based on PageRank
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 12:02:49 AM, Nov 11, 2016
 *
 */

public class PRVote extends ScoredVotingRule {
	public int maxIter = 50;

	public <T> ScoredItems<T> getScoredRanking(Profile<T> profile) {
		T[] items = profile.getSortedItems();
		double[][] scores = getPairMarginScores(profile, items);
		int numItem = items.length;
		// probabilistic transportation matrix
		double[][] a = new double[numItem][numItem];
		for (int i = 0; i < numItem; i++) {
			double sum = 0;
			for (int j = 0; j < numItem; j++) {
				a[j][i] = 1.0 / (1 + Math.exp(-scores[j][i]));
				sum += a[j][i];
			}
			for (int j = 0; j < numItem; j++)
				a[j][i] /= sum;
		}
		return new ScoredItems<>(items, computePR(a));
	}

	<T> double[][] getPairMarginScores(Profile<T> profile, T[] items) {
		int numItem = items.length;

		double[][] marginTable = new double[numItem][numItem];
		int index = -1;
		for (T[] pref : profile.data) {
			index++;
			for (int i = 0; i < numItem; i++) {
				int runner = Arrays.binarySearch(items, pref[i]);
				for (int j = i + 1; j < numItem; j++) {
					int opponent = Arrays.binarySearch(items, pref[j]);
					float val = (j - i) * (1.0f / (i + 1)) * profile.votes[index];
					marginTable[runner][opponent] += val;
					marginTable[opponent][runner] -= val;
				}
			}
		}
		return marginTable;
	}

	double[] computePR(double[][] a) {
		int n = a.length;
		double[] p = MathLib.Rand.distribution(n);
		for (int t = 0; t < maxIter; t++) {
			double[] pp = MathLib.Matrix.ax(a, p);
			MathLib.Scale.sum(pp);
			p = Arrays.copyOf(pp, n);
		}
		return p;
	}

	public static void main(String[] args) {
		Eval1 eval = new Eval1();
		int[] numVotes = MathLib.Series.range(5, 2, 15);
		VotingRule oracle = new Borda();
		
		StringBuffer sb = new StringBuffer();
		sb.append("nv,condorcet,pairmargin,pagerankvote\n");
		for (int i = 0; i < numVotes.length; i++) {
			int nv = numVotes[i];
			sb.append(nv + ",");
			sb.append(eval.getSimilarity(3, nv, oracle, new Condorcet()) + ",");
			sb.append(eval.getSimilarity(3, nv, oracle, new PairMargin()) + ",");
			sb.append(eval.getSimilarity(3, nv, oracle, new PRVote()) + "\n");
		}
		System.out.println(sb.toString());
	}

	public static void main_(String[] args) throws Exception {
		TickClock.beginTick();

		List<VotingAxiom> axioms = new ArrayList<>();
		axioms.add(new CondorcetCriterion());
		axioms.add(new CondorcetLoserCriterion());
		axioms.add(new ConsistencyCriterion());
		axioms.add(new HomogeneityCriterion());
		axioms.add(new MajorityCriterion());
		axioms.add(new MajorityLoserCriterion());
		axioms.add(new MonotonicityCriterion());
		axioms.add(new NeutralityCriterion());
		axioms.add(new ParticipationCriterion());
		axioms.add(new PluralityCriterion());
		axioms.add(new ReversalSymmetryCriterion());
		axioms.add(new SmithCriterion());
		axioms.add(new SchwarzCriterion());

		// PRVote rule = new PRVote();
		VotingRule rule = new Borda();

		int numItem = 3;
		List<Callable<double[]>> tasks = new ArrayList<>();
		int num = 8;
		for (int i = 0; i < num; i++) {
			int nv = 2 * i + 5;
			tasks.add(() -> getSAT(numItem, nv, rule, axioms));
		}
		List<double[]> results = MulticoreExecutor.run(tasks);

		StringBuffer sb = new StringBuffer();
		for (int c = 0; c < axioms.size(); c++) {
			sb.append(axioms.get(c).toString());
			for (int i = 0; i < num; i++)
				sb.append("," + results.get(i)[c]);
			sb.append("\n");
		}

		OpenOption[] options = new OpenOption[] { StandardOpenOption.CREATE, StandardOpenOption.WRITE };
		Files.write(Paths.get("./csc/sat-prvote.txt"), sb.toString().getBytes(), options);

		TickClock.stopTick();
	}

	public static double[] getSAT(int numItem, int numVote, VotingRule rule, List<VotingAxiom> axioms) {
		List<Profile<Integer>> profiles = null;
		profiles = DataEngine.getAECPreferenceProfiles(numItem, numVote).collect(Collectors.toList());

		int n = axioms.size();
		double[] sat = new double[n];
		for (int i = 0; i < n; i++)
			sat[i] = axioms.get(i).getSatisfiability(profiles, rule);
		System.out.println(Arrays.toString(sat));
		return sat;
	}
}
