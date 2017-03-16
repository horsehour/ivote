package com.horsehour.vote.rule.multiseat;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Position include the position value and all alternatives who are placed at
 * the position. It is very crucial trait to construct the preference rankings
 * for all three kinds of ordering: strictly complete ordering, partial ordering
 * and complete ordering with ties.
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 10:05:21 PM, Dec 22, 2016
 *
 */
public class Position {
	public int val;
	public List<Integer> items;

	public Position(int val, List<Integer> items) {
		this.val = val;
		this.items = new ArrayList<>(items);
	}

	public Position(int val, int item) {
		this.val = val;
		this.items = Arrays.asList(item);
	}

	/**
	 * Moving up (d < 0) or down (d > 0). The smaller the value is, the higher
	 * the position is.
	 * 
	 * @param d
	 */
	public void move(int d) {
		this.val += d;
	}

	public String toString() {
		return val + " : " + items;
	}
}