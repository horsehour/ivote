package com.horsehour.vote.data;

import java.io.IOException;
import java.nio.file.OpenOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.horsehour.util.TickClock;

/**
 * Data Set from Voting Profiles
 * 
 * @author Chunheng Jiang
 * @since Mar 24, 2017
 * @version 1.0
 */
public class VoteSet {
	static OpenOption[] options = { StandardOpenOption.APPEND, StandardOpenOption.CREATE, StandardOpenOption.WRITE };

	public double[][] features;
	public int[][] winners;

	/**
	 * whether encoding winners in terms of binary vector with k hotting points
	 */
	public boolean hotEncoding = true;
	/**
	 * number of candidates
	 */
	public int m;
	/**
	 * dimension of the extracted input features
	 */
	public int d;

	public VoteSet(double[][] features, int[][] winners, boolean hotEncoding) {
		if (features == null || winners == null) {
			System.err.println("Null input(s) and output(s).");
			System.exit(-1);
		}

		int n = features.length;
		if (n != winners.length) {
			System.out.println("Inconsistent input(s) and output(s).");
			return;
		}

		this.features = Arrays.copyOf(features, n);
		this.winners = Arrays.copyOf(winners, n);
		this.hotEncoding = hotEncoding;
		if (hotEncoding)
			this.m = winners[0].length;
	}

	/**
	 * Encode the winners in binary vector
	 * 
	 * @param k
	 */
	public void encode(int k) {
		if (hotEncoding) {
			System.err.println("It has been encoded.");
			return;
		}

		int n = winners.length;
		int[] previous = null;
		for (int i = 0; i < n; i++) {
			int len = winners[i].length;
			if (len > k)
				System.err.println("Inconsistent number of winners.");

			previous = Arrays.copyOf(winners[i], len);
			winners[i] = new int[k];
			for (int j = 0; j < len; j++) {
				int ind = previous[j];
				winners[i][ind] = 1;
			}
		}

		hotEncoding = true;
		m = k;
	}

	/**
	 * Decode the winners from binary vector
	 */
	public void decode() {
		if (!hotEncoding) {
			System.err.println("It hasn't been encoded.");
			return;
		}

		int n = winners.length;
		List<Integer> winnerList = null;
		for (int i = 0; i < n; i++) {
			winnerList = new ArrayList<>();
			for (int j = 0; j < m; j++) {
				if (winners[i][j] == 1)
					winnerList.add(j);
			}

			int len = winnerList.size();
			winners[i] = new int[len];
			for (int j = 0; j < len; j++)
				winners[i][j] = winnerList.get(j);
		}
		hotEncoding = false;
	}

	/**
	 * Flatten the inputs and outputs to ensure each input with a single winner
	 * 
	 * @param inputs
	 * @param outputs
	 */
	public void flatten(List<double[]> inputs, List<Integer> outputs) {
		inputs = new ArrayList<>();
		outputs = new ArrayList<>();

		int n = features.length;
		if (hotEncoding) {
			for (int i = 0; i < n; i++) {
				double[] input = features[i];
				for (int j = 0; j < m; j++) {
					if (winners[i][j] == 1) {
						inputs.add(input);
						outputs.add(j);
					}
				}
			}
		} else {
			for (int i = 0; i < n; i++) {
				double[] input = features[i];
				for (int w : winners[i]) {
					inputs.add(input);
					outputs.add(w);
				}
			}
		}
	}

	public static void main(String[] args) throws IOException {
		TickClock.beginTick();

		TickClock.stopTick();
	}
}