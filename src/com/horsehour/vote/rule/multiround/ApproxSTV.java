package com.horsehour.vote.rule.multiround;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.OpenOption;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import org.apache.commons.lang3.SerializationUtils;

import com.horsehour.ml.metric.MAP;
import com.horsehour.util.MathLib;
import com.horsehour.util.TickClock;
import com.horsehour.vote.Profile;
import com.horsehour.vote.data.DataEngine;
import com.horsehour.vote.data.FeatureLab;
import com.horsehour.vote.data.VoteExp;

import smile.classification.SoftClassifier;

/**
 * Approximation to multi-round STV by aggregating the predictions from top-k
 * layers. We do not need to compute the plurality score for the kth layer.
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 3:06:45 AM, Jan 10, 2017
 */
public class ApproxSTV {
	public int k = 2;
	public int numItemTotal;
	public int numNode;
	public Node root;

	public Profile<Integer> profile;
	public SoftClassifier<double[]> algo;

	public List<Integer> predictions;
	public Set<List<Integer>> visited;

	public int rank = 0;

	public String base = "/users/chjiang/github/csc/";
	public String dataset = "soc-3";

	public VoteExp exp;

	public ApproxSTV() {}

	public class Node {
		public List<Integer> state;
		public int depth = 0;

		public List<Node> children;

		public Node(List<Integer> candidates) {
			this.state = new ArrayList<>(candidates);
		}

		public Node(List<Integer> candidates, Integer eliminated) {
			this.state = new ArrayList<>(candidates);
			this.state.remove(eliminated);
		}
	}

	/**
	 * Make prediction using the learned algorithm
	 * 
	 * @param profile
	 * @param items
	 * @return prediction scores over all alternatives
	 */
	List<Double> predict(Profile<Integer> profile, List<Integer> items) {
		List<Double> p = new ArrayList<>(Collections.nCopies(numItemTotal, 0.0));
		if (items.size() == 1) {
			p.set(items.get(0), 1.0);
			return p;
		}

		double[] posteriori = new double[numItemTotal];
		double[] features = FeatureLab.getF1(profile, items, numItemTotal);
		algo.predict(features, posteriori);

		// adjust the predictions such that those not included are assigned zero
		for (int i = 0; i < numItemTotal; i++) {
			int ind = items.indexOf(i);
			if (ind == -1)
				posteriori[i] = 0;
		}

		MathLib.Scale.sum(posteriori);
		for (int i = 0; i < posteriori.length; i++)
			p.add(posteriori[i]);
		return p;
	}

	/**
	 * Calculate plurality score from a preference profile with those unstated
	 * candidates eliminated
	 * 
	 * @param profile
	 * @param state
	 * @return Plurality score of these candidates
	 */
	public int[] scoring(Profile<Integer> profile, List<Integer> state) {
		int[] votes = profile.votes;
		int[] scores = new int[state.size()];
		int c = 0;
		for (Integer[] pref : profile.data) {
			for (int i = 0; i < pref.length; i++) {
				int item = pref[i];
				int index = state.indexOf(item);
				/** item has been eliminated **/
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
	 * Remove candidates who have never been the first choice of any voter and
	 * reconstruct the profile
	 * 
	 * @param profile
	 * @param state
	 * @return reconstructed profile if there are any candidate has being
	 *         eliminated, elsewise return the original profile
	 */
	public Profile<Integer> preprocess(Profile<Integer> profile, List<Integer> state) {
		int[] scores = scoring(profile, state);
		List<Integer> eliminated = new ArrayList<>();
		for (int i = 0; i < scores.length; i++) {
			if (scores[i] == 0)
				eliminated.add(i);
		}

		if (eliminated.isEmpty())
			return profile;
		else {
			state.removeAll(eliminated);
			return profile.reconstruct(state);
		}
	}

	void elect() {
		LinkedList<Node> fringe = new LinkedList<>();
		fringe.add(root);

		Node next = null;
		while (!fringe.isEmpty()) {
			next = fringe.pollLast();
			List<Integer> state = next.state;
			if (next.depth == k || visited.contains(state))
				continue;

			numNode++;
			double[] features = FeatureLab.getF1(profile, state, numItemTotal);
			predictions.addAll(exp.mechanism.apply(features, state));

			int[] scores = scoring(profile, state);
			next.children = new ArrayList<>();
			int[] min = MathLib.argmin(scores);
			for (int i : min) {
				Node child = new Node(state, state.get(i));
				child.depth = next.depth + 1;
				next.children.add(child);
			}
			fringe.addAll(next.children);
		}
	}

	/**
	 * @param profile
	 * @return All possible winners with various tie-breaking rules
	 */
	public List<Integer> getAllWinners(Profile<Integer> profile) {
		this.numNode = 0;
		this.visited = new HashSet<>();
		this.predictions = new ArrayList<>();

		Integer[] items = profile.getSortedItems();
		List<Integer> state = new ArrayList<>(Arrays.asList(items));
		this.profile = preprocess(profile, state);
		this.root = new Node(state);
		this.elect();
		// TODO: select those with the highest frequency being predicted as
		// winner
		return predictions.stream().distinct().sorted().collect(Collectors.toList());
	}

	public Map<Integer, Long> getWinners(Profile<Integer> profile) {
		this.numNode = 0;
		this.visited = new HashSet<>();
		this.predictions = new ArrayList<>();

		Integer[] items = profile.getSortedItems();
		List<Integer> state = new ArrayList<>(Arrays.asList(items));
		this.profile = preprocess(profile, state);
		this.root = new Node(state);
		this.elect();
		return MathLib.Data.count(predictions);
	}

	/**
	 * Compute three performance indicators: Precision, Recall and F-score
	 * 
	 * @param truth
	 * @param pred
	 * @return three performance measures
	 */
	public float[] getPerformance(List<List<Integer>> truth, List<List<Integer>> pred) {
		int n = truth.size(), nWinner = 0, nElected = 0, nHit = 0;
		List<Integer> winners = null, elected = null;
		for (int i = 0; i < n; i++) {
			winners = truth.get(i);
			elected = pred.get(i);

			if (winners == null || winners.isEmpty() || elected == null || elected.isEmpty())
				continue;

			nWinner += winners.size();
			nElected += elected.size();

			for (int elect : elected) {
				if (winners.contains(elect))
					nHit++;
			}
		}

		float[] perf = new float[3];
		perf[0] = nHit * 1.0f / nElected;
		perf[1] = nHit * 1.0f / nWinner;
		perf[2] = 2 * perf[0] * perf[1] / (perf[0] + perf[1]);
		return perf;
	}

	public static void main(String[] args) throws IOException {
		TickClock.beginTick();
		String base = "/users/chjiang/github/csc/";
		String dataset = "soc-3-hardcase";

		ApproxSTV rule = new ApproxSTV();

		rule.numItemTotal = 10;
		rule.exp = new VoteExp();
		rule.exp.m = rule.numItemTotal;
		rule.exp.cutoff = 0.5;

		Path model = Paths.get(base + "logistic.mdl");
		rule.algo = SerializationUtils.deserialize(Files.readAllBytes(model));
		rule.exp.algo = rule.algo;

		List<List<Integer>> t = null;
		List<List<Integer>> p = null;

		Profile<Integer> profile;
		Map<Integer, Long> pred;

		OpenOption[] options = { StandardOpenOption.APPEND, StandardOpenOption.CREATE, StandardOpenOption.WRITE };
		StringBuffer sb = null;

		MAP map = new MAP();
		List<String> lines = Files.readAllLines(Paths.get(base + "winners-stv-soc3.txt"));
		for (String line : lines) {
			String[] cells = line.split("\t");
			String name = cells[0];

			int m = Integer.parseInt(name.substring(1, name.indexOf("N")));
			if (m > 10)
				continue;

			System.out.println(name);
			
			if (!Files.exists(Paths.get(base + dataset + "/" + name)))
				profile = DataEngine.loadProfile(Paths.get(base + "soc-3" + "/" + name));
			else
				profile = DataEngine.loadProfile(Paths.get(base + dataset + "/" + name));

			String label = cells[1];
			cells = label.split(",");
			List<Integer> truth = new ArrayList<>();
			for (int i = 0; i < cells.length; i++) {
				if (cells[i].contains("1"))
					truth.add(i);
			}
			t = new ArrayList<>();
			t.add(truth);

			pred = rule.getWinners(profile);
			List<Integer> predict = null;
			predict = new ArrayList<>(pred.keySet());
			// predict = rule.getAllWinners(profile);
			p = new ArrayList<>();
			p.add(predict);
			float[] perf = rule.getPerformance(t, p);
			
			truth.clear();
			predict.clear();
			for(int i = 0; i < cells.length; i++){
				truth.add(Integer.parseInt(cells[i]));
				predict.add(0);
			}

			for(int i : pred.keySet()){
				predict.set(i, pred.get(i).intValue());
			}

			sb = new StringBuffer();
			sb.append(name).append("\t").append(rule.numNode).append("\t");
			sb.append(truth).append("\t").append(predict).append("\t");
			sb.append(Arrays.toString(perf)).append("\t");
			sb.append(map.measure(truth, predict)).append("\n");

			Files.write(Paths.get(base + "approx.pred.csv"), sb.toString().getBytes(), options);
		}
		TickClock.stopTick();
	}
}
