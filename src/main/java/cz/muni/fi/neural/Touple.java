package cz.muni.fi.neural;

/**
 * Created by ondrejvelisek on 4.12.16.
 */
public class Touple<T, S> {

	private T first;
	private S second;

	public Touple(T first, S second) {
		this.first = first;
		this.second = second;
	}

	public T getFirst() {
		return first;
	}

	public void setFirst(T first) {
		this.first = first;
	}

	public S getSecond() {
		return second;
	}

	public void setSecond(S second) {
		this.second = second;
	}

	@Override
	public boolean equals(Object o) {
		if (this == o) return true;
		if (o == null || getClass() != o.getClass()) return false;

		Touple<?, ?> touple = (Touple<?, ?>) o;

		if (first != null ? !first.equals(touple.first) : touple.first != null) return false;
		return second != null ? second.equals(touple.second) : touple.second == null;

	}

	@Override
	public int hashCode() {
		int result = first != null ? first.hashCode() : 0;
		result = 31 * result + (second != null ? second.hashCode() : 0);
		return result;
	}
}
