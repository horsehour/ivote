package com.horsehour.vote.axiom;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BiPredicate;
import java.util.stream.Stream;

import org.apache.commons.lang3.ArrayUtils;

import com.horsehour.util.TickClock;
import com.horsehour.vote.ChoiceTriple;
import com.horsehour.vote.Profile;
import com.horsehour.vote.rule.Schulze;
import com.horsehour.vote.rule.VotingRule;

/**
 * The monotonicity criterion is a voting system criterion used to analyze both
 * single and multiple winner voting systems. A voting system is monotonic if it
 * satisfies one of the definitions of the monotonicity criterion, given below.
 * <p>
 * Douglas R. Woodall, calling the criterion mono-raise, defines it as: A
 * candidate x should not be harmed [i.e., change from being a winner to a
 * loser] if x is raised on some ballots without changing the orders of the
 * other candidates.
 * <p>
 * Note that the references to orders and relative positions concern the
 * rankings of candidates other than X, on the set of ballots where X has been
 * raised. So, if changing a set of ballots voting "A > B > C" to "B > C > A"
 * causes B to lose, this does not constitute failure of Monotonicity, because
 * in addition to raising B, we changed the relative positions of A and C. This
 * criterion may be intuitively justified by reasoning that in any fair voting
 * system, no vote for a candidate, or increase in the candidate's ranking,
 * should instead hurt the candidate. It is a property considered in Arrow's
 * impossibility theorem. Some political scientists, however, doubt the value of
 * monotonicity as an evaluative measure of voting systems. David Austen-Smith
 * and Jeffrey Banks, for example, published an article in The American
 * Political Science Review in which they argue that "monotonicity in electoral
 * systems is a nonissue: depending on the behavioral model governing individual
 * decision making, either everything is monotonic or nothing is monotonic."
 * <p>
 * Although all voting systems are vulnerable to tactical voting, systems which
 * fail the monotonicity criterion suffer an unusual form, where voters with
 * enough information about other voter strategies could theoretically try to
 * elect their candidate by counter-intuitively voting against that candidate.
 * Tactical voting in this way presents an obvious risk if a voter's information
 * about other ballots is wrong, however, and there is no evidence that voters
 * actually pursue such counter-intuitive strategies in non-monotonic voting
 * systems in real-world elections.
 * <p>
 * Of the single-winner voting systems, plurality voting (first past the post),
 * Borda count, Schulze method, and Ranked Pairs (Maximize Affirmed Majorities)
 * are monotonic, while Coombs' method, runoff voting and instant-runoff voting
 * are not. The single-winner methods of range voting, majority judgment and
 * approval voting are also monotonic as one can never help a candidate by
 * reducing or removing support for them, but these require a slightly different
 * definition of monotonicity as they are not preferential systems.
 * <p>
 * Of the multiple-winner voting systems, all plurality voting methods are
 * monotonic, such as plurality-at-large voting (bloc voting), cumulative
 * voting, and the single non-transferable vote. Most versions of the single
 * transferable vote, including all variants currently in use for public
 * elections (which simplify to instant runoff when there is only one winner)
 * are not monotonic.
 * <p>
 * A voting system is monotonic, if no winner is harmed by up-ranking in a
 * voter's preference. To evaluate whether a voting rule r satisfies the
 * monotonicity criterion, we select a preference profile P = (R1, R2, ..., Rn),
 * where Ri means a preference ranking of m candidates by voter vi, n is the
 * number of voters. According to the rule, a winner c is selected, i.e. r(P) =
 * c. For all voters, one voter raises the position of c in his/her preference
 * ranking, while the other voters' preference profiles are not changed, if the
 * rule still select c as the winner. Therefore, we could say that the rule
 * satisfies the monotonicity criterion. Each voter have many possible changes
 * of the position of c in his/her preference ranking, the quantity is
 * straightforward, equal to (m - 1) - b(c, R). Here, m is the number of
 * candidates, b(c, p) is the Borda score c receives from R. If c stands on the
 * first place of R, m - 1 points will be assigned. When ranked second, m - 2
 * points, et. al. Suppose c is the most favorite candidate in R, the
 * corresponding voter will have no chance to raise c's ranking in R, since b(c,
 * R) = m - 1. The satisfiability of a voting rule to the monotonicity could be
 * computed by dividing the possible number of rank-raising that produces a
 * different winner to the total number of rank-raising performed on all the
 * preference rankings in the profile P - sum{R: (m - 1) - b(c, R)} = n (m - 1)
 * - b(c, P). Here, b(c, P) is the Borda score the candidate c on profile P.
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 10:56:14 AM, Jun 21, 2016
 *
 */
public class MonotonicityCriterion extends VotingAxiom {
	BiPredicate<Integer, List<Integer>> violator = (winner, predicted) -> {
		if (predicted == null || !predicted.contains(winner))
			return true;
		else
			return false;
	};

	/**
	 * Approximately evaluate the satisfiability with sampling profiles
	 * 
	 * @param profiles
	 * @param rule
	 * @return approximated satisfiability
	 */
	public double getSatisfiability(Stream<Profile<Integer>> profiles, VotingRule rule) {
		AtomicLong numTotal = new AtomicLong(0), numMatch = new AtomicLong(0);
		profiles.forEach(profile -> {

			List<Integer> winners = rule.getAllWinners(profile);
			if (winners == null)
				return;

			// long stat = profile.getStat();
			long[] count = raiseAndEvalMatch(profile, winners, rule);
			numTotal.addAndGet(count[0]);
			numMatch.addAndGet(count[1]);
		});
		return numMatch.doubleValue() / numTotal.doubleValue();
	}

	public boolean isViolated(Stream<Profile<Integer>> profiles, VotingRule rule) {
		return profiles.anyMatch(profile -> {
			List<Integer> winners = rule.getAllWinners(profile);
			if (winners == null)
				return false;

			return raiseAndEvalViolation(profile, winners, rule);
		});
	}

	/**
	 * Suppose one voter changes his/her mind, raise the rank of one winner in
	 * his/her preference ranking list. If the action changes the outcome, and
	 * one of the winners drops out. It will contribute one point to the
	 * unsatisfiability of the rule to the monotonicity criterion. For
	 * multi-winner, each winner should be examined by each changes in each
	 * preference ranking if the raising is available.
	 * 
	 * @param profile
	 * @param winners
	 * @param rule
	 * @return total number of changes and the number of matches
	 */
	long[] raiseAndEvalMatch(Profile<Integer> profile, List<Integer> winners, VotingRule rule) {
		// number of different preferences
		int numPref = profile.votes.length;
		int numItem = profile.getNumItem();

		AtomicLong numTotal = new AtomicLong(0), numMatch = new AtomicLong(0);
		for (int winner : winners) {
			for (int i = 0; i < numPref; i++) {
				int j = ArrayUtils.indexOf(profile.data[i], winner);

				Integer[] preference = null;
				int[] numPrefList = null;

				Integer[][] preferences = Arrays.stream(profile.data).map(a -> a.clone()).toArray(Integer[][]::new);
				if (profile.votes[i] > 1) {
					preferences = ArrayUtils.add(preferences, Arrays.copyOf(profile.data[i], numItem));
					numPrefList = Arrays.copyOf(profile.votes, numPref + 1);
					numPrefList[i] -= 1;
					numPrefList[numPref] = 1;
					preference = preferences[numPref];
				} else {
					numPrefList = Arrays.copyOf(profile.votes, numPref);
					preference = preferences[i];
				}

				// ranking raising will be performed on this preference
				for (; j > 0; j--) {
					preference[j] = preference[j - 1];
					preference[j - 1] = winner;

					numTotal.addAndGet(profile.votes[i]);

					Profile<Integer> raisedProfile = new Profile<>(preferences, numPrefList);
					List<Integer> raisedWinners = rule.getAllWinners(raisedProfile);
					if (!violator.test(winner, raisedWinners))
						// all these preferences are the same
						numMatch.addAndGet(profile.votes[i]);
				}
			}
		}
		return new long[] { numTotal.longValue(), numMatch.longValue() };
	}

	boolean raiseAndEvalViolation(Profile<Integer> profile, List<Integer> winners, VotingRule rule) {
		// number of different preferences
		int numPref = profile.votes.length;
		int numItem = profile.getNumItem();

		for (int winner : winners) {
			for (int i = 0; i < numPref; i++) {
				int j = ArrayUtils.indexOf(profile.data[i], winner);

				Integer[] preference = null;
				int[] numPrefList = null;

				Integer[][] preferences = Arrays.stream(profile.data).map(a -> a.clone()).toArray(Integer[][]::new);
				if (profile.votes[i] > 1) {
					preferences = ArrayUtils.add(preferences, Arrays.copyOf(profile.data[i], numItem));
					numPrefList = Arrays.copyOf(profile.votes, numPref + 1);
					numPrefList[i] -= 1;
					numPrefList[numPref] = 1;
					preference = preferences[numPref];
				} else {
					numPrefList = Arrays.copyOf(profile.votes, numPref);
					preference = preferences[i];
				}

				// ranking raising will be performed on this preference
				for (; j > 0; j--) {
					preference[j] = preference[j - 1];
					preference[j - 1] = winner;

					Profile<Integer> raisedProfile = new Profile<>(preferences, numPrefList);
					List<Integer> raisedWinners = rule.getAllWinners(raisedProfile);
					if (violator.test(winner, raisedWinners))
						return true;
				}
			}
		}
		return false;
	}

	public List<ChoiceTriple<Integer>> getAllRaisedProfiles(Profile<Integer> profile, List<Integer> winners) {
		List<ChoiceTriple<Integer>> result = new ArrayList<>();

		// number of different preferences
		int numPref = profile.votes.length;
		int numItem = profile.getNumItem();

		for (int winner : winners) {

			for (int i = 0; i < numPref; i++) {
				int j = ArrayUtils.indexOf(profile.data[i], winner);

				Integer[] preference = null;
				int[] numPrefList = null;

				Integer[][] preferences = Arrays.stream(profile.data).map(a -> a.clone()).toArray(Integer[][]::new);
				if (profile.votes[i] > 1) {
					preferences = ArrayUtils.add(preferences, Arrays.copyOf(profile.data[i], numItem));
					numPrefList = Arrays.copyOf(profile.votes, numPref + 1);
					numPrefList[i] -= 1;
					numPrefList[numPref] = 1;
					preference = preferences[numPref];
				} else {
					numPrefList = Arrays.copyOf(profile.votes, numPref);
					preference = preferences[i];
				}

				// ranking raising will be performed on this preference
				for (; j > 0; j--) {
					preference[j] = preference[j - 1];
					preference[j - 1] = winner;

					Profile<Integer> raisedProfile = new Profile<>(preferences, numPrefList);
					result.add(new ChoiceTriple<>(raisedProfile, Arrays.asList(winner), profile.votes[i]));
				}
			}
		}
		result.add(new ChoiceTriple<>(profile, winners, 1));
		return result;
	}

	public static void main(String[] args) {
		TickClock.beginTick();

		VotingAxiom axiom = null;
		 axiom = new MonotonicityCriterion();
//		axiom = new ParticipationCriterion();

		System.out.println(axiom.getSatisfiability(3, 9, new Schulze()));

		TickClock.stopTick();

	}
}