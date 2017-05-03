package com.horsehour.vote.axiom;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BiPredicate;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.apache.commons.lang3.tuple.Pair;

import com.horsehour.util.TickClock;
import com.horsehour.vote.Profile;
import com.horsehour.vote.data.DataEngine;
import com.horsehour.vote.rule.Bucklin;
import com.horsehour.vote.rule.VotingRule;

/**
 * Participation criterion is also known as Bram and Fishburn's
 * "no show paradox". Suppose that a voting rule select x as a winner. Suppose
 * there is a ballot where candidate x is strictly preferred to candidate y, a
 * paradox occurs if the addition of the ballot to the original profile makes
 * the winner changing from x to y. Then, we conclude that the voting rule
 * violates the participation paradox. Participation criterion encourages voters
 * to cast a sincere ballots rather than stay at home not voting at all.
 * <p>
 * The criterion does not require that x must still win, only that adding voters
 * who do not prefer y should not make y the winner. We note that the
 * participation criterion is different from the monotonicity principle.
 * Monotonicity states that among existing voters, a change in voter preferences
 * that favors x against y must not hurt x’s position in the final decision.
 * <p>
 * Facts about participation:
 * <li>Condorcet criterion implies participation criterion</li>
 * <li>Scoring rules satisfy participation criterion</li>
 * <li>All Condorcet consistent methods violates participation criterion among
 * four or more candidates</li>
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 3:36:59 PM, Jul 29, 2016
 */

public class ParticipationCriterion extends VotingAxiom {

	BiPredicate<Integer, List<Integer>> violator = (winner, predicted) -> {
		if (predicted == null || !predicted.contains(winner))
			return true;
		else
			return false;
	};

	int numItem, numVote;

	@Override
	public double getSatisfiability(int numItem, int numVote, VotingRule rule) {
		List<Profile<Integer>> profiles = new ArrayList<>();
		for (int nv = 1; nv <= numVote; nv++)
			profiles.addAll(DataEngine.getAECPreferenceProfiles(numItem, nv).collect(Collectors.toList()));

		this.numItem = numItem;
		this.numVote = numVote;

		return super.getSatisfiability(profiles, rule);
	}

	@Override
	public double getSatisfiability(List<Profile<Integer>> profiles, VotingRule rule) {
		this.numItem = profiles.get(0).getNumItem();
		this.numVote = profiles.stream().map(profile -> profile.numVoteTotal).max(Integer::max).get();

		return super.getSatisfiability(profiles, rule);
	}

	public double getSatisfiability(Stream<Profile<Integer>> profiles, VotingRule rule) {
		AtomicLong numTotal = new AtomicLong(0), numMatch = new AtomicLong(0);
		/**
		 * (Single winner, List of Profiles)
		 */
		Map<Integer, List<Profile<Integer>>> cluster = profiles.map(profile -> {
			List<Integer> winners = rule.getAllWinners(profile);
			if (winners == null || winners.size() > 1)
				return null;
			else
				return Pair.of(profile, winners.get(0));
		}).filter(Objects::nonNull).collect(Collectors.groupingBy(pair -> pair.getRight(),
				Collectors.mapping(pair -> pair.getLeft(), Collectors.toList())));

		List<List<Integer>> permutations = DataEngine.getAllPermutations(numItem);
		cluster.keySet().stream().forEach(winner -> {
			Stream<List<Integer>> preferences = permutations.stream().filter(list -> list.get(0) == winner);
			List<Profile<Integer>> hesitantBallots = getHesitantBallotList(preferences, numVote);
			Pair<AtomicLong, AtomicLong> count = showAndEvalMatch(cluster.get(winner), winner, hesitantBallots, rule);
			numTotal.addAndGet(count.getLeft().get());
			numMatch.addAndGet(count.getRight().get());
		});
		return numMatch.doubleValue() / numTotal.doubleValue();
	}

	/**
	 * 
	 * @param preferences
	 * @param maxNumVote
	 * @return construct all possible hesistant ballots those prefer current
	 *         winners to other candidates
	 */
	<T> List<Profile<T>> getHesitantBallotList(Stream<List<T>> preferences, int maxNumVote) {
		List<Profile<T>> hesitantBallots = new ArrayList<>();
		preferences.forEach(preference -> {
			List<List<T>> data = new ArrayList<>();
			data.add(preference);
			for (int i = 1; i <= maxNumVote; i++)
				hesitantBallots.add(new Profile<>(data, new int[] { i }));
		});
		return hesitantBallots;
	}

	Pair<AtomicLong, AtomicLong> showAndEvalMatch(List<Profile<Integer>> profiles, int winner,
			List<Profile<Integer>> hesitantBallots, VotingRule rule) {
		AtomicLong numTotal = new AtomicLong(0), numMatch = new AtomicLong(0);

		for (Profile<Integer> profile : profiles)
			for (Profile<Integer> unsureBallot : hesitantBallots) {
				numTotal.addAndGet(1);

				Profile<Integer> profileShown = profile.merge(unsureBallot);
				List<Integer> winnersShown = rule.getAllWinners(profileShown);

				if (!violator.test(winner, winnersShown))
					numMatch.addAndGet(1);
			}
		return Pair.of(numTotal, numMatch);
	}

	@Override
	public boolean isViolated(int numItem, int numVote, VotingRule rule) {
		List<Profile<Integer>> profiles = new ArrayList<>();
		for (int nv = 1; nv <= numVote; nv++)
			profiles.addAll(DataEngine.getAECPreferenceProfiles(numItem, nv).collect(Collectors.toList()));
		this.numItem = numItem;
		this.numVote = numVote;

		return super.isViolated(profiles, rule);
	}

	@Override
	public boolean isViolated(List<Profile<Integer>> profiles, VotingRule rule) {
		this.numItem = profiles.get(0).getNumItem();
		this.numVote = profiles.stream().map(profile -> profile.numVoteTotal).max(Integer::max).get();

		return super.isViolated(profiles, rule);
	}

	public boolean isViolated(Stream<Profile<Integer>> profiles, VotingRule rule) {
		/**
		 * (Single winner, List of Profiles)
		 */
		Map<Integer, List<Profile<Integer>>> cluster = profiles.map(profile -> {
			List<Integer> winners = rule.getAllWinners(profile);
			if (winners == null || winners.size() > 1)
				return null;
			else
				return Pair.of(profile, winners.get(0));
		}).filter(Objects::nonNull).collect(Collectors.groupingBy(pair -> pair.getRight(),
				Collectors.mapping(pair -> pair.getLeft(), Collectors.toList())));

		List<List<Integer>> permutations = DataEngine.getAllPermutations(numItem);
		return cluster.keySet().stream().anyMatch(winner -> {
			Stream<List<Integer>> preferences = permutations.stream().filter(list -> list.get(0) == winner);
			List<Profile<Integer>> hesitantBallots = getHesitantBallotList(preferences, numVote);
			return discourageParticipation(cluster.get(winner), winner, hesitantBallots, rule);
		});
	}

	boolean discourageParticipation(List<Profile<Integer>> profiles, int winner, List<Profile<Integer>> hesitantBallots,
			VotingRule rule) {
		for (Profile<Integer> profile : profiles)
			for (Profile<Integer> unsureBallot : hesitantBallots) {
				Profile<Integer> profileShown = profile.merge(unsureBallot);
				List<Integer> winnersShown = rule.getAllWinners(profileShown);
				if (violator.test(winner, winnersShown))
					return true;
			}
		return false;
	}

	public static void main(String[] args) {
		TickClock.beginTick();

		int numItem = 3, numVote = 7;

		VotingRule rule = null;
		// rule = new Plurality<>();
		// rule = new Borda<>();
		// rule = new Copeland<>();
		// rule = new InstantRunoff<>();
		// rule = new PairMarginRule<>();
		// rule = new RankedPairs<>();
		rule = new Bucklin();

		ParticipationCriterion axiom = new ParticipationCriterion();
		System.out.println(axiom.getSatisfiability(numItem, numVote, rule));

		TickClock.stopTick();
	}
}