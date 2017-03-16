package com.horsehour.vote;

import java.util.List;

import org.apache.commons.lang3.tuple.Triple;

/**
 *
 * @author Chunheng Jiang
 * @version 1.0
 * @since 1:06:57 PM, Aug 15, 2016
 *
 */

public class ChoiceTriple<T>{
	Triple<Profile<T>, List<T>, Integer> triple;
	
	public ChoiceTriple(Profile<T> profile, List<T> winners, int num){
		triple = Triple.of(profile, winners, num);
	}

	public Profile<T> getProfile() {
		return triple.getLeft();
	}

	public List<T> getWinners() {
		return triple.getMiddle();
	}
	
	public T getWinner(){
		return triple.getMiddle().get(0);
	}

	public int size() {
		return triple.getRight();
	}
}