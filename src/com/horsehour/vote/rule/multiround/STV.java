package com.horsehour.vote.rule.multiround;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

import com.horsehour.util.TickClock;
import com.horsehour.vote.Profile;
import com.horsehour.vote.data.DataEngine;

/**
 * Single transferable voting method (STV, a.k.a Cincinnati rule or Hare voting)
 * eliminates no more than one candidate at a time. STV with single winner is
 * sometimes called "alternative voting". When there are multiple winners, the
 * algorithm for STV becomes considerably complex. The implementation tries to
 * have all the winners based on each possible tie-breaking rule.
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 1:10:41 PM, Dec 21, 2016
 */
public class STV {
	public int numItemTotal;
	public Set<List<Integer>> visited;

	public Set<Integer> winners;
	public List<Integer> trace;

	public Node root;
	public long begin;

	public int numNode;

	public STV() {}

	public class Node {
		public int priority = 0;
		public int[] scores;
		public List<Integer> state;

		public Node(List<Integer> candidates) {
			this.state = new ArrayList<>(candidates);
		}

		public Node(List<Integer> candidates, Integer eliminated) {
			this.state = new ArrayList<>(candidates);
			this.state.remove(eliminated);
		}
	}

	/**
	 * Calculate plurality score from a preference profile with those unstated
	 * candidates eliminated
	 * 
	 * @param profile
	 * @param candidates
	 * @return Plurality score of specific candidates
	 */
	public int[] getPluralityScore(Profile<Integer> profile, List<Integer> candidates) {
		int[] votes = profile.votes;
		int[] scores = new int[candidates.size()];

		int c = 0;
		for (Integer[] pref : profile.data) {
			for (int i = 0; i < pref.length; i++) {
				int item = pref[i];
				int index = candidates.indexOf(item);
				// item has been eliminated
				if (index == -1)
					continue;
				scores[index] += votes[c];
				break;
			}
			c++;
		}
		return scores;
	}

	/**
	 * @param remaining
	 * @return Evaluate whether all remaining alternatives have been elected as
	 *         the winners
	 */
	boolean foundAll(List<Integer> remaining) {
		for (int item : remaining)
			if (!winners.contains(item))
				return false;
		return true;
	}

	/**
	 * Remove candidates who have never been the first choice of any voter and
	 * reconstruct the profile
	 * 
	 * @param profile
	 * @param candidates
	 * @return reconstructed profile if there are any candidate has being
	 *         eliminated, elsewise return the original profile
	 */
	Profile<Integer> preprocess(Profile<Integer> profile, List<Integer> candidates) {
		int[] scores = getPluralityScore(profile, candidates);
		List<Integer> eliminated = new ArrayList<>();
		for (int i = 0; i < scores.length; i++) {
			if (scores[i] == 0)
				eliminated.add(i);
		}
		if (eliminated.isEmpty())
			return profile;

		candidates.removeAll(eliminated);
		return profile.reconstruct(candidates);
	}

	/**
	 * @param profile
	 * @return All possible winners with various tie-breaking rules
	 */
	public List<Integer> getAllWinners(Profile<Integer> profile) {
		visited = new HashSet<>();
		trace = new ArrayList<>();
		winners = new HashSet<>();

		numNode = 0;

		Integer[] items = profile.getSortedItems();

		root = new Node(Arrays.asList(items));
		profile = preprocess(profile, root.state);

		LinkedList<Node> fringe = new LinkedList<>();
		fringe.add(root);

		Node top = null;
		while (!fringe.isEmpty()) {
			top = fringe.pollLast();
			List<Integer> candidates = top.state;
			int size = candidates.size();
			int item = candidates.get(0);
			if ((size == 1) && (!winners.contains(item))) {
				numNode++;
				winners.add(item);
				trace.add(numNode);
				visited.add(candidates);
				continue;
			}

			if (foundAll(candidates)){
				continue;
			}

			int[] scores = getPluralityScore(profile, candidates);
			int min = profile.numVoteTotal;
			for (int score : scores) {
				if (score < min)
					min = score;
			}

			for (int i = 0; i < candidates.size(); i++) {
				if (scores[i] == min) {
					int tied = candidates.get(i);
					Node child = new Node(candidates, tied);
					if(visited.contains(child.state))
						continue;
					fringe.add(child);
				}
			}
			numNode++;
			visited.add(candidates);
		}
		
		List<Integer> ret = new ArrayList<>();
		for (int winner : winners)
			ret.add(winner);
		Collections.sort(ret);
		return ret;
	}

	public static void main(String[] args) throws IOException {
		// M40N70-39, M50N70-23, M30N30-237, M30N30-238
		TickClock.beginTick();

		String base = "/Users/chjiang/Github/csc/";
		String dataset = "m40n40-stv";

		Path input = Paths.get(base + dataset + "/M40N40-100024.csv");
		STV rule = new STV();

		Profile<Integer> profile = DataEngine.loadProfile(input);
		List<Integer> winners = rule.getAllWinners(profile);

		String format = "winners=%s, trace=%s, node=%d\n";
		System.out.printf(format, winners, rule.trace, rule.numNode);

		TickClock.stopTick();
	}
}