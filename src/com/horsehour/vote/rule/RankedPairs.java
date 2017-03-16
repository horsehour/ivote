package com.horsehour.vote.rule;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.lang3.tuple.Triple;
import org.jgrapht.alg.CycleDetector;
import org.jgrapht.graph.DefaultDirectedGraph;
import org.jgrapht.graph.DefaultEdge;

import com.horsehour.util.TickClock;
import com.horsehour.vote.Profile;
import com.horsehour.vote.axiom.MonotonicityCriterion;

/**
 * Ranked pairs (RP) a.k.a the Tideman method is named after its creator
 * Nicolaus Tideman. As a Condorcet method, it is built upon pairwise
 * comparison, requires three steps to determine a winner. The first step is to
 * tally the vote count comparing each pair of candidates and determine the
 * winner in each pairwise competition. The tally adds one point to each
 * preferred candidate in pair comparison. The second step is to sort pairs
 * according to the points that its victor gets. When two different victors get
 * the same points, then the one which loses less will be ranked first. The last
 * step is called lock. Lock step examines the pair based on the sorted list,
 * and connects one pair's winner to its loser unless the pair makes a cycle in
 * the graph. Based on the principle, a directed acyclic graph of candidates is
 * created. The source vertex of the graph is the final winner.
 * 
 * <p>
 * Edges with negative weights are not considered, because up till that, the
 * winner can be figured out using only all the positive weights.
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 6:02:22 PM, Jul 27, 2016
 */

public class RankedPairs extends Condorcet {

	@Override
	public <T> List<T> getRanking(Profile<T> profile) {
		T[] items = profile.getSortedItems();
		int[][] ppm = getPairwisePreferenceMatrix(profile, items);

		int nItem = items.length;
		float[] winnings = new float[nItem];
		List<Triple<Integer, Integer, Integer>> pairList = new ArrayList<>();

		for (int i = 0; i < nItem; i++) {
			for (int j = i + 1; j < nItem; j++) {
				if (ppm[i][j] > ppm[j][i]) {
					winnings[i]++;
					pairList.add(Triple.of(i, j, ppm[i][j]));
				} else if (ppm[i][j] == ppm[j][i]) {// even # voters
					winnings[i] += 0.5F;
					winnings[j] += 0.5F;
					pairList.add(Triple.of(i, j, ppm[i][j]));
					pairList.add(Triple.of(j, i, ppm[j][i]));
				} else {
					winnings[j]++;
					pairList.add(Triple.of(j, i, ppm[j][i]));
				}
			}
		}

		/**
		 * ranking pairs
		 */
		pairList.sort((t1, t2) -> {
			/**
			 * rank pairs according to pairs' strengths
			 */
			int ret = t2.getRight().compareTo(t1.getRight());
			if (ret == 0) {
				if (t2.getLeft() == t1.getLeft())// same winner
					/**
					 * ranking pairs according to losers' winning scores
					 */
					ret = Float.compare(winnings[t2.getMiddle()], winnings[t1.getMiddle()]);
				else
					/**
					 * ranking pairs according to winners' winning scores
					 */
					ret = Float.compare(winnings[t2.getLeft()], winnings[t1.getLeft()]);
			}
			return ret;
		});

		List<Integer> ranking = lockIn(pairList, nItem);
		List<T> socialPreference = new ArrayList<>();
		for (int i : ranking)
			socialPreference.add(items[i]);
		return socialPreference;
	}

	@Override
	public <T> List<T> getAllWinners(Profile<T> profile) {
		return Arrays.asList(getRanking(profile).get(0));
	}

	List<Integer> lockIn(List<Triple<Integer, Integer, Integer>> pairList, int nItem) {
		DefaultDirectedGraph<Integer, DefaultEdge> graph = new DefaultDirectedGraph<>(DefaultEdge.class);
		for (int i = 0; i < nItem; i++)
			graph.addVertex(i);

		CycleDetector<Integer, DefaultEdge> detector = null;

		/**
		 * adding the highest two pairs to the direct graph no need to check the
		 * existence of a cycle
		 */
		for (int i = 0; i < pairList.size(); i++) {
			Triple<Integer, Integer, Integer> triple = pairList.get(i);
			graph.addEdge(triple.getLeft(), triple.getMiddle());

			if (i > 1) {
				detector = new CycleDetector<>(graph);
				if (detector.detectCycles())
					graph.removeEdge(triple.getLeft(), triple.getRight());
			}
		}

		List<Integer> ranking = new ArrayList<>();
		int[] degrees = new int[graph.vertexSet().size()];
		int index = 0;
		for (int vertex : graph.vertexSet()) {
			ranking.add(index);
			degrees[index++] = graph.outDegreeOf(vertex);
		}

		/**
		 * source vertex is the node with a largest out-degree
		 */
		ranking.sort((i, j) -> Integer.compare(degrees[j], degrees[i]));
		return ranking;
	}

	public static void main(String[] args) {
		TickClock.beginTick();

		RankedPairs rule = new RankedPairs();
		// NeutralityCriterion criterion = new NeutralityCriterion();
		MonotonicityCriterion criterion = new MonotonicityCriterion();
		// ConsistencyCriterion criterion = new ConsistencyCriterion();

		int numItem = 3, numVote = 5;

		System.out.println(criterion.getSatisfiability(numItem, numVote, rule));

		TickClock.stopTick();
	}
}