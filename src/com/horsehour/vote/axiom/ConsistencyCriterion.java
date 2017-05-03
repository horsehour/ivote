package com.horsehour.vote.axiom;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.Callable;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BiPredicate;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.apache.commons.lang3.tuple.Pair;

import com.horsehour.util.MathLib;
import com.horsehour.util.MulticoreExecutor;
import com.horsehour.util.TickClock;
import com.horsehour.vote.Profile;
import com.horsehour.vote.data.DataEngine;
import com.horsehour.vote.rule.InstantRunoff;
import com.horsehour.vote.rule.VotingRule;

/**
 * A voting system is consistent if, when the electorate is divided arbitrarily
 * into two (or more) parts and separate elections in each part result in the
 * same choice being selected, an election of the entire electorate also selects
 * that alternative. Smith calls this property separability and Woodall calls it
 * convexity.
 * <p>
 * It has been proven a preferential voting system is consistent if and only if
 * it is a positional voting system. Borda count is an example of this. The
 * failure of the consistency criterion can be seen as an example of Simpson's
 * paradox.
 * <p>
 * Smith and Young characterized scoring rules via four axioms: consistency,
 * continuity, anonymity and neutrality.
 * <p>
 * 
 * Consistency requires of a voting rule that if two disjoint electrorates yield
 * outcomes with some common features, then the rule applied to the combined
 * electrorate also yields an outcome with these features.
 * <p>
 * 
 * Continuity expresses the idea that sufficient small changes in the expressed
 * preference of the voters can not change a loser into a winner.
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 1:04:20 AM, Jun 28, 2016
 *
 */

public class ConsistencyCriterion extends VotingAxiom {
	/**
	 * Combined profiles should select the same winners of the individual
	 * profiles
	 */
	BiPredicate<Integer, List<Integer>> violator = (winner, predicted) -> {
		if (predicted == null || predicted.size() > 1)
			return true;
		if (predicted.get(0) == winner)
			return false;
		else
			return true;
	};

	/**
	 * Evaluate all possible combinations of the preferences in the full
	 * preference space. Each evaluation should contain two subset, we can
	 * demonstrate the possible layout of the two subsets in terms of their
	 * sizes. All possible duo-size patterns could be listed as following: <br>
	 * (1,1)<br>
	 * (1,2), (2,2)<br>
	 * (1,3), (2,3), (3,3)<br>
	 * (1,4), (2,4), (3,4), (4,4)<br>
	 * ..., ..., ..., ..., ... <br>
	 * (1,n), (2,n), (3,n), (4,n), ..., (n-1, n), (n, n)
	 * <p>
	 * For pattern (1,1), there are m! * m! = m!^2 possible combinations of
	 * preference profiles. Pattern (1,2) contains m!*(m!^2) = m!^3. Therefore,
	 * the total number of the possible combinations of preference profiles is
	 * extremely high, it's \sum_{i=1}^n \sum_{i<=j<=n} (m!)^{i+j}.
	 * 
	 * <p>
	 * When m and n become larger, it will be impossible to emunerate all
	 * preference profile combinations. Then, it would be intractable to
	 * evaluate the satisfiability to the consistency criterion for a voting
	 * rule.
	 * <p>
	 * The solution is random sampling. For example, we could use the accept-
	 * rejection sampling method to approximate the exact satisfiability.
	 * According to the sampling method, each possible preference ranking
	 * represents an axis, the feasible domain for samping is an m!-dimensional
	 * non-negative real space. The rejection sampling method could produce
	 * diversity samples, make the evaluation result close to the exact
	 * satisfiability.
	 */
	@Override
	public double getSatisfiability(int numItem, int numVote, VotingRule rule) {
		List<Profile<Integer>> profiles = new ArrayList<>();
		for (int votes = 1; votes <= numVote; votes++)
			profiles.addAll(DataEngine.getAECPreferenceProfiles(numItem, votes).collect(Collectors.toList()));
		return super.getSatisfiability(profiles, rule);
	}

	public double getSatisfiability(Stream<Profile<Integer>> profiles, VotingRule rule) {
		Map<Integer, List<Profile<Integer>>> cluster = getValidPremises(profiles, rule);

		AtomicLong numTotal = new AtomicLong(0), numMatch = new AtomicLong(0);

		List<MatchTask> taskList = new ArrayList<>(cluster.size());
		for (int winner : cluster.keySet())
			taskList.add(new MatchTask(cluster.get(winner), winner, rule));

		try {
			for (long[] count : MulticoreExecutor.run(taskList)) {
				numTotal.addAndGet(count[0]);
				numMatch.addAndGet(count[1]);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return numMatch.doubleValue() / numTotal.doubleValue();
	}

	/**
	 * Construct valid premises
	 * 
	 * @param profiles
	 * @param rule
	 * @return valid premises to evaluate consistency criterion
	 */
	<T> Map<T, List<Profile<T>>> getValidPremises(Stream<Profile<T>> profiles, VotingRule rule) {
		Stream<Pair<T, Profile<T>>> stream = profiles.flatMap(profile -> {
			List<T> winners = rule.getAllWinners(profile);
			// just for single winner
			if (winners == null || winners.size() > 1)
				return null;
			return winners.stream().map(winner -> Pair.of(winner, profile));
		}).filter(Objects::nonNull);

		Map<T, List<Profile<T>>> premises = stream.collect(
				Collectors.groupingBy(p -> p.getLeft(), Collectors.mapping(p -> p.getRight(), Collectors.toList())));
		return premises;
	}

	@Override
	public boolean isViolated(int numItem, int numVote, VotingRule rule) {
		List<Profile<Integer>> profiles = new ArrayList<>();
		for (int votes = 1; votes <= numVote; votes++)
			profiles.addAll(DataEngine.getAECPreferenceProfiles(numItem, votes).collect(Collectors.toList()));
		return isViolated(profiles, rule);
	}

	@Override
	public boolean isViolated(Stream<Profile<Integer>> profiles, VotingRule rule) {
		Map<Integer, List<Profile<Integer>>> cluster = getValidPremises(profiles, rule);
		List<ViolationTask> taskList = new ArrayList<>(cluster.size());
		for (int winner : cluster.keySet())
			taskList.add(new ViolationTask(cluster.get(winner), winner, rule));

		try {
			return MulticoreExecutor.run(taskList).stream().anyMatch(ret -> ret);
		} catch (Exception e) {
			e.printStackTrace();
			return false;
		}
	}

	class MatchTask implements Callable<long[]> {
		List<Profile<Integer>> profiles;
		VotingRule rule;
		int winner;

		MatchTask(List<Profile<Integer>> profiles, int winner, VotingRule rule) {
			this.profiles = profiles;
			this.winner = winner;
			this.rule = rule;
		}

		/**
		 * combine two individual profiles and from which the rule selects one
		 * winner. If the selected winner is the same as it selects from the two
		 * individual profiles, then the rule will cumulative one point for its
		 * satisfiability to the consistency criterion
		 */
		public long[] call() throws Exception {
			long numTotal = 0, numMatch = 0;
			Profile<Integer> unionProfile = null;

			for (int i = 0; i < profiles.size(); i++) {
				for (int j = i; j < profiles.size(); j++) {
					numTotal++;

					if (j == i)
						unionProfile = new Profile<>(profiles.get(i).data, MathLib.Matrix.multiply(profiles.get(i).votes, 2));
					else
						unionProfile = profiles.get(i).merge(profiles.get(j));

					List<Integer> predicted = rule.getAllWinners(unionProfile);
					if (!violator.test(winner, predicted))
						numMatch++;
				}
			}
			return new long[] { numTotal, numMatch };
		}
	}

	class ViolationTask implements Callable<Boolean> {
		List<Profile<Integer>> profiles;
		VotingRule rule;
		int winner;

		ViolationTask(List<Profile<Integer>> profiles, int winner, VotingRule rule) {
			this.profiles = profiles;
			this.winner = winner;
			this.rule = rule;
		}

		/**
		 * combine two individual profiles and from which the rule selects one
		 * winner. If the selected winner is the same as it selects from the two
		 * individual profiles, then the rule will cumulative one point for its
		 * satisfiability to the consistency criterion
		 */
		public Boolean call() throws Exception {
			Profile<Integer> unionProfile = null;

			for (int i = 0; i < profiles.size(); i++) {
				for (int j = i; j < profiles.size(); j++) {
					if (j == i)
						unionProfile = new Profile<>(profiles.get(i).data, MathLib.Matrix.multiply(profiles.get(i).votes, 2));
					else
						unionProfile = profiles.get(i).merge(profiles.get(j));

					List<Integer> predict = rule.getAllWinners(unionProfile);
					if (violator.test(winner, predict))
						return true;
				}
			}
			return false;
		}
	}

	public static void main(String[] args) {
		TickClock.beginTick();

		int numItem = 3, numVote = 5;

		VotingRule rule = null;
		// rule = new Maximin();
		// rule = new Plurality();
		// rule = new Borda();
		// rule = new Copeland();
		rule = new InstantRunoff();
		// rule = new PairMarginRule();
		// rule = new RankedPairs();
		// rule = new Condorcet();

		ConsistencyCriterion axiom = new ConsistencyCriterion();

		System.out.println(axiom.isViolated(numItem, numVote, rule));

		TickClock.stopTick();
	}
}