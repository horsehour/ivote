package com.horsehour.vote.stat;

import jnr.ffi.byref.DoubleByReference;
import jnr.ffi.byref.IntByReference;

import com.sun.jna.Library;

/**
 * Interface-mapped library to mvnpack.so
 * @author mao
 *
 */
public interface MvnPackGenz extends Library {

	public static final String MVNPACK_SO = "mvnpack.so";

	/**
	 * See http://www.math.wsu.edu/faculty/genz/software/fort77/mvtdstpack.f
	 * @param n
	 * @param lower
	 * @param upper
	 * @param infin
	 * @param correl
	 * @param maxpts
	 * @param abseps
	 * @param releps
	 * @param error
	 * @param value
	 * @param inform
	 */
	void mvndst_(IntByReference n, double[] lower, double[] upper, int[] infin, double[] correl, IntByReference maxpts, DoubleByReference abseps,
	        DoubleByReference releps, DoubleByReference error, DoubleByReference value, IntByReference inform);

	/**
	 * See http://www.math.wsu.edu/faculty/genz/software/fort77/mvnexppack.f
	 * @param n
	 * @param lower
	 * @param upper
	 * @param infin
	 * @param correl
	 * @param maxpts
	 * @param abseps
	 * @param releps
	 * @param error
	 * @param value
	 * @param inform
	 */
	void mvnexp_(IntByReference n, double[] lower, double[] upper, int[] infin, double[] correl, IntByReference maxpts, DoubleByReference abseps,
	        DoubleByReference releps, double[] error, double[] value, IntByReference inform);

	/**
	 * See http://www.math.wsu.edu/faculty/genz/software/fort77/mvnxpppack.f
	 * @param n
	 * @param lower
	 * @param upper
	 * @param infin
	 * @param correl
	 * @param maxpts
	 * @param abseps
	 * @param releps
	 * @param error
	 * @param value
	 * @param inform
	 */
	void mvnxpp_(IntByReference n, double[] lower, double[] upper, int[] infin, double[] correl, IntByReference maxpts, DoubleByReference abseps,
	        DoubleByReference releps, double[] error, double[] value, IntByReference inform);

}
