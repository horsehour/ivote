package com.horsehour.vote;

/**
 *
 * @author Chunheng Jiang
 * @version 1.0
 * @since 11:42:41 PM, Dec 20, 2016
 *
 */

public class Candidate {
	public int id;
	public int pos;

	public Candidate(int id, int pos){
		this.id = id;
		this.pos = pos;
	}

	public String toString(){
		return id + " : " + pos;
	}
}
