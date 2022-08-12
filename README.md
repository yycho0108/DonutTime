## Donut Time

Script for IM^2 lab donut time, discussion group assignment.
Automatically tracks history.
Takes the best result between MINLP solution and monte carlo simulation.

## How to run

### IpOpt installation
First, install `ipopt` binary and put it somewhere in your path.
One way to do so is as follows:
```bash
$ wget https://www.coin-or.org/download/binary/Ipopt/Ipopt-3.7.1-linux-x86_64-gcc4.3.2.tgz
$ dtrx Ipopt-3.7.1-linux-x86_64-gcc4.3.2.tgz
```

### Running the Demo

Then,

```bash
$ ./opt.sh 

num people 17
num teams 3
dt=1, weight=1.0
dt=2, weight=0.5
Objective value:
99.5
==week 00 == 
	== room [current room] == 
	[['changjae', 'dongchan', 'junhyek', 'jaehyung', 'junyeob']]
	== room [imcube] == 
	[['beomjoon', 'dongwon', 'heesang', 'jisu', 'minchan']]
	== room [elsewhere in the universe] == 
	[['dongryung', 'jamie', 'jiyong', 'quang-minh', 'sanghyeon', 'wonjae', 'yoonwoo']]
==week 01 == 
	== room [current room] == 
	[['beomjoon', 'dongchan', 'jaehyung', 'jamie', 'junyeob']]
	== room [imcube] == 
	[['dongryung', 'dongwon', 'jiyong', 'minchan', 'quang-minh']]
	== room [elsewhere in the universe] == 
	[['changjae', 'junhyek', 'heesang', 'jisu', 'sanghyeon', 'wonjae', 'yoonwoo']]
==week 02 == 
	== room [current room] == 
	[['changjae', 'junhyek', 'heesang', 'jamie', 'minchan', 'quang-minh']]
	== room [imcube] == 
	[['beomjoon', 'dongchan', 'dongryung', 'dongwon', 'jisu']]
	== room [elsewhere in the universe] == 
	[['jaehyung', 'jiyong', 'junyeob', 'sanghyeon', 'wonjae', 'yoonwoo']]
```
