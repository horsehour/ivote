package com.horsehour.vote.axiom;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Stack;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BiPredicate;
import java.util.stream.Stream;

import com.horsehour.util.TickClock;
import com.horsehour.vote.Profile;
import com.horsehour.vote.rule.Condorcet;
import com.horsehour.vote.rule.RankedPairs;
import com.horsehour.vote.rule.VotingRule;

/**
 * The Smith criterion (sometimes generalized Condorcet criterion, after John H.
 * Smith) is a voting systems criterion defined such that its satisfaction by a
 * voting system occurs when the system always picks the winner from the
 * "Smith set", the smallest non-empty set of candidates such that every member
 * of the set is pairwise preferred to every candidate not in the set. One
 * candidate is pairwise preferred over another candidate if, in a one-on-one
 * competition, more voters prefer the first candidate than prefer the other
 * candidate.
 * <p>
 * The Smith set is named for mathematician John H Smith, whose version of the
 * Condorcet criterion is actually stronger than that defined for social welfare
 * functions. Benjamin Ward was probably the first to write about this set,
 * which he called the "majority set".
 * <p>
 * The Smith criterion implies the Condorcet criterion, since if there is a
 * Condorcet winner, then that winner is the only member of the Smith set.
 * Obviously, this means that failing the Condorcet criterion automatically
 * implies the non-compliance with the Smith criterion as well.
 * <p>
 * Additionally, such sets comply with the Condorcet loser criterion. This is
 * notable, because even some Condorcet methods do not (Minimax). It also
 * implies the mutual majority criterion, since the Smith set is a subset of the
 * "mutual majority set". The Smith criterion implies the Mutual majority
 * criterion, therefore Minimax' failure to the Mutual majority criterion is
 * also a failure to the Smith criterion.
 * <p>
 * The "Sincere Smith set" is the basis for a definition of the Smith Criterion
 * that is applicable to all voting systems. Its definition is that
 * "X is socially preferred to Y if more voters prefer X to Y than prefer Y to X"
 * . It's the smallest set of candidates such that every candidate in the set is
 * socially preferred to every candidate outside the set. Based on the
 * definition of sincere smith set, we have the definition of Smith Criterion:
 * If everyone votes sincerely, the winner should come from the sincere Smith
 * set. The Smith set represents one approach to identifying a subset of
 * candidates from which an election method should choose a winner. The idea is
 * to choose the "best compromise."
 * <p>
 * The Smith set can be calculated with the Floyd–Warshall algorithm in time Θ
 * (n^3). It can also be calculated using a version of Kosaraju's algorithm or
 * Tarjan's algorithm in time Θ (n^2).
 * <p>
 * Two other names for the Smith set are the Top Cycle and the GETCHA set.
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 8:26:57 PM, Jul 31, 2016
 *
 */

public class SmithCriterion extends VotingAxiom {
	/**
	 * Prediction is not in Smith set or Schwarz set
	 */
	BiPredicate<List<?>, List<?>> violator = (set, predicted) -> {
		if (predicted == null)
			return true;
		return predicted.stream().anyMatch(p -> !set.contains(p));
	};

	@Override
	public double getSatisfiability(Stream<Profile<Integer>> profiles, VotingRule rule) {
		AtomicLong numTotalCP = new AtomicLong(0), numMatch = new AtomicLong(0);
		profiles.forEach(profile -> {
			List<Integer> smithSet = getSmithSet(profile);
			if (smithSet == null)
				return;

			// long stat = profile.getStat();
			numTotalCP.addAndGet(1);

			List<Integer> predicted = rule.getAllWinners(profile);
			if (!violator.test(smithSet, predicted))
				numMatch.addAndGet(1);
		});
		return numMatch.doubleValue() / numTotalCP.doubleValue();
	}

	@Override
	public boolean isViolated(Stream<Profile<Integer>> profiles, VotingRule rule) {
		return profiles.anyMatch(profile -> {
			List<Integer> smithSet = getSmithSet(profile);
			if (smithSet == null)
				return false;

			List<Integer> predicted = rule.getAllWinners(profile);
			return violator.test(smithSet, predicted);
		});
	}

	int numItem;

	<T> List<T> getSmithSet(Profile<T> profile) {
		T[] sortedItems = profile.getSortedItems();
		numItem = sortedItems.length;
		int[][] ppm = Condorcet.getPairwisePreferenceMatrix(profile, sortedItems);
		boolean[][] relation = new boolean[numItem][numItem];
		for (int i = 0; i < numItem; i++)
			for (int j = 0; j < numItem; j++)
				relation[i][j] = ppm[i][j] > ppm[j][i];
		return getMaximalCandidates(sortedItems, relation);
	}

	<T> List<T> getMaximalCandidates(T[] sortedItems, boolean[][] relation) {
		numItem = sortedItems.length;
		searchSCCKosaraju(relation);
		List<T> maximalCandidates = new ArrayList<>();

		for (int i = 0; i < sortedItems.length; i++)
			if (scc[i] == 0)// maximal scc
				maximalCandidates.add(sortedItems[i]);
		return maximalCandidates;
	}

	/**
	 * Search strongly connected components in a directed graph using the
	 * Kosaraju two-way algorithm
	 * 
	 * @param relation
	 */
	void searchSCCKosaraju(boolean[][] relation) {
		int[] searchOrder = new int[numItem];
		for (int i = 0; i < numItem; i++)
			searchOrder[i] = i;

		dfs(relation, searchOrder);

		/**
		 * reversal searching order
		 */
		int[] searchOrderR = new int[numItem];
		boolean[][] relationT = new boolean[numItem][numItem];
		for (int i = 0; i < numItem; i++) {
			searchOrderR[i] = stack.pop();
			for (int j = 0; j < numItem; j++)
				relationT[i][j] = relation[j][i];
		}

		dfs(relationT, searchOrderR);
	}

	Stack<Integer> stack;
	boolean[] visited;
	int[] scc;
	int indexSCC;

	/**
	 * prepare for dfs
	 */
	void prepareDFS() {
		stack = new Stack<>();
		visited = new boolean[numItem];
		Arrays.fill(visited, false);

		// Strongly Connected Components (SCCs)
		indexSCC = 0;
		scc = new int[numItem];
	}

	/**
	 * depth-first searching
	 * 
	 * @param relation
	 * @param searchOrder
	 */
	void dfs(boolean[][] relation, int[] searchOrder) {
		prepareDFS();
		for (int index : searchOrder) {
			if (!visited[index]) {
				search(relation, searchOrder, index);
				indexSCC++;
			}
		}
	}

	/**
	 * travesal graph
	 * 
	 * @param relation
	 * @param searchOrder
	 * @param indexFrom
	 */
	void search(boolean[][] relation, int[] searchOrder, int indexFrom) {
		scc[indexFrom] = indexSCC;
		visited[indexFrom] = true;

		for (int indexTo : searchOrder)
			if (relation[indexFrom][indexTo]) {
				if (!visited[indexTo])
					search(relation, searchOrder, indexTo);
			}

		/**
		 * all vertices reachable have been processed
		 */
		stack.push(indexFrom);
	}

	public static void main(String[] args) {
		TickClock.beginTick();

		VotingAxiom axiom = new SmithCriterion();
		VotingRule rule = null;
		rule = new RankedPairs();
		// rule = new Nanson();
		// rule = new Schulze();
		// rule = new Borda();
		// rule = new Condorcet();

		System.out.println(axiom.getSatisfiability(3, 5, rule));

		TickClock.stopTick();
	}
}
