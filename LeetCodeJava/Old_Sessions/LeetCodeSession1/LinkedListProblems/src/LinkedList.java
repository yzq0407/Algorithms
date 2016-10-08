
public class LinkedList {
	Node head;
	int size;
	
	public static class Node {
		int data;
		Node next;
		Node(){
			
		}
		
		Node(int data) {
			this.data = data;
			this.next = null;
		}
	}
	public LinkedList() {
		head = null;
		size = 0;
	}
	
	public LinkedList(int data){
		head = new Node(data);
		size = 1;
	}
	
	public void append (int data) {
		if (head != null){
			Node current = head;
			while (current.next!=null){
				current = current.next;
			}
			current.next = new Node(data);
			size ++;
			
		}
		else{
			head  = new Node(data);
			size = 1;
		}
		
	}
	
	public void deleteTail(){
		if (size > 1) {
			Node current = head;
			while(current.next.next!=null){
				current = current.next;
			}
			current.next = null;
		}
		else {
			head = null;
		}
		size = (size-1>=0)?size-1:size;
	}
	
	public void printAllData(){
		if (head!=null){
			System.out.print(head.data);
			Node current = head.next;
			while (current!=null){
				System.out.print(" --> "+current.data);
				current = current.next;
			}
			System.out.println("");
		}
	}
	
	public static void main(String[] args) {
		LinkedList ls = new LinkedList (5);
		ls.append(6);
		ls.append(5);
		ls.append(4);
		ls.append(9);
		ls.append(11);
		ls.append(2);
		ls.append(28);
		ls.printAllData();
		System.out.println(ls.size);
		while(ls.size>0){
			ls.deleteTail();
			ls.printAllData();
			System.out.println(ls.size);
		}
		ls.append(6);
		ls.append(5);
		ls.append(4);
		ls.append(9);
		ls.append(11);
		ls.append(2);
		ls.append(28);
		ls.printAllData();
		
	}
	
	
	

}
