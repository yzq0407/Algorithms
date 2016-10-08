import java.util.Arrays;


public class Problems {
	public static void rearrange(LinkedList ls){
		if (ls.size<2 ||ls.size%2!=0)
			throw new IllegalArgumentException();
		LinkedList.Node p2 = ls.head.next;
		LinkedList.Node p1 = ls.head.next;
		while (p1.next!=null) {
			p1 = p1.next.next;
			p2 = p2.next;
		}
		p1 = ls.head;
		while (p2.next!=null){
			LinkedList.Node temp1 = p1.next;
			LinkedList.Node temp2 = p2.next;
			p1.next = p2;
			p2.next = temp1;
			p1 = temp1;
			p2 = temp2;
		}
		p1.next = p2;
		p2.next = null;
	}
	
	public static  void partitionLL(LinkedList ls, Integer par) {
		//check if there is any node equal to partition
		LinkedList.Node current = ls.head;
		boolean isInList = false;
		while (current.next!=null) {
			if (current.next.data == par){
				isInList = true;
				current = current.next;
				break;
			}
				
			else
				current = current.next;
		}
		if (!isInList){
			LinkedList.Node beforeHead = null;
			LinkedList.Node beforeTail = null;
			LinkedList.Node afterHead = null;
			LinkedList.Node afterTail = null;
			LinkedList.Node p1 = ls.head;
			while (p1!=null) {
				if ((int)p1.data<par){
					if (beforeHead==null){
						beforeHead = p1;
						beforeTail = p1;
					}
					else {
						beforeTail.next = p1;
						beforeTail = beforeTail.next;
					}
				}
				else {
					if (afterHead ==null) {
						afterHead = p1;
						afterTail = p1;
					}
					else{
						afterTail.next = p1;
						afterTail = afterTail.next;
					}
				}
				p1 = p1.next;
			}
			beforeTail.next = afterHead;
			afterTail = null;
			ls.head = beforeHead;
			
		}
	}
	
	
	
	public static void inverseLinkedList(LinkedList ls){
		if (ls.size==1)
			return;
		else if (ls.size ==2) {
			LinkedList.Node temp = ls.head;
			ls.head = temp.next;
			ls.head.next = temp;
		}
		else {
			LinkedList.Node p1 = ls.head;
			LinkedList.Node p2 = p1.next;
			LinkedList.Node p3 = p2.next;
			p1.next = null;
			while (p3!=null) {
				p2.next = p1;
				p1 = p2;
				p2 = p3;
				p3 = p3.next;				
			}
			p2.next = p1;
			ls.head = p2;
			
		}
			
	}
	
	public static int[] returnLastKelements (LinkedList ls, int k){
		int[] result = new int[k];
		returnLastKelements (ls.head, result);
		return result;
		
		
	}
	
	public static LinkedList SumLinkedListReverse(LinkedList A, LinkedList B){
		LinkedList.Node Anode = A.head;
		LinkedList.Node Bnode = B.head;
		LinkedList result = new LinkedList();
		LinkedList.Node head = new LinkedList.Node();
		LinkedList.Node p = head;
		int oneMoreDigit = 0;
		while (Anode!=null&Bnode!=null) {
			int sum = (int)Anode.data + (int)Bnode.data+oneMoreDigit;
			oneMoreDigit = (sum>=10)?1:0;
			sum = sum - oneMoreDigit*10;
			p.next = new LinkedList.Node(sum);
			p = p.next;
			Anode = Anode.next;
			Bnode = Bnode.next;
		}
		if (Anode ==null)
			copyRest(Bnode, p, oneMoreDigit);
		else
			copyRest(Anode, p, oneMoreDigit);
		result.head = head.next;
		return result;
	}
	
	public static void copyRest(LinkedList.Node restNode, LinkedList.Node resultNode, int digit) {
		while (restNode!=null) {
			LinkedList.Node last = new LinkedList.Node();
			int sum = restNode.data+digit;
			digit = (sum>=10)?1:0;
			sum  = sum-digit*10;
			last.data = sum;
			restNode = restNode.next;
			resultNode.next = last;
			resultNode = resultNode.next;
		}
		if (digit!=0){
			LinkedList.Node last = new LinkedList.Node(digit);
			resultNode.next = last;
		}
	}
	
	
//	private int sumTwoNodes (LinkedList.Node nodeA, LinkedList.Node nodeB, LinkedList.Node result){
//		if (nodeA.next==null&&nodeB.next==null){
//			LinkedList.Node node = new LinkedList.Node();
//			int sum = (int)nodeA.data + (int)nodeB.data;
//			if (sum>=10) {
//				node.data = sum-10;
//				node.next = result;
//				return 1;
//			}
//		}
//		
//	}
	
	private static int returnLastKelements(LinkedList.Node node, int [] result) {
		if (node.next==null){
			result[result.length-1] = (int)node.data;
			return 2;
		}
		else {
			int offset = returnLastKelements(node.next, result);
			if (offset > result.length){
				return offset;
			}
			else {
				result[result.length-offset] = (int)node.data;
				return offset+1;
			}
		}
	}
	
	public static void main (String[] args) {
		LinkedList ls1 = new LinkedList (7);
		ls1.append(1);
		ls1.append(6);
		LinkedList ls2 = new LinkedList (5);
		ls2.append(9);
		ls2.append(2);
		ls2.append(1);
		ls2.append(3);
//		ls.append(28);
		ls1.printAllData();
		ls2.printAllData();
		SumLinkedListReverse(ls1, ls2).printAllData();
//		System.out.println(Arrays.toString(returnLastKelements(ls , 9)));
//		ls.printAllData();
	}

}
