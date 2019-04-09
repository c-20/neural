#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

typedef struct _arr {
  double *v;
  int len;
} arr;

typedef struct _node {
  arr weights;
  double output, delta;
} node;

typedef struct _layer {
  node *nodes;
  int numnodes;
  arr output;
} layer;

typedef struct _network {
  layer layers[3]; // later layer *layers;
  int numlayers;
  arr output;
} network;

double sigmoid(double value) {
  return 1.0 / (1.0 + exp(-1.0 * value));
}

double calculateoutputvalue(arr inputs, node *n) {
  if (inputs.len != n->weights.len - 1)
    { printf("mismatch %d != %d - 1.", inputs.len, n->weights.len); return -9999.0; }
  double activation = 0.0;
  int wi = -1;
  while (++wi < inputs.len)
    { activation += n->weights.v[wi + 1] * inputs.v[wi]; }
  activation += n->weights.v[0];
  n->output = sigmoid(activation);
  return n->output; // set output and return it
}

#define LEARNINGCOEFFICIENT  0.3
#define NETWORKNODES         10

layer initlayer(int numinputs, int numnodes) {
  layer l;
  l.nodes = (node *)calloc(numnodes, sizeof(node));
  l.numnodes = numnodes;
  if (numnodes == 1) { // input layer (1 node) starts with 0.0 weights
    // input layer has no extra weight at 0
    l.nodes[0].weights.v = (double *)calloc(numinputs, sizeof(double));
    int vi = -1;
    while (++vi < numinputs)
      { l.nodes[0].weights.v[vi] = 0.0; }
    l.nodes[0].weights.len = numinputs; // no extra weight at [0]
    l.nodes[0].output = 0.0;
    l.nodes[0].delta = 0.0;
    l.output.v = (double *)calloc(numinputs, sizeof(double));
    l.output.len = numinputs;
  } else {  // backpropagation layers (n nodes) start with random weights
    srand(time(NULL));
    int ni = -1;
    while (++ni < numnodes) {
      l.nodes[ni].weights.v = (double *)calloc(numinputs + 1, sizeof(double));
      l.nodes[ni].weights.len = numinputs + 1;
      int ii = -1;
      while (++ii < l.nodes[ni].weights.len) {
        double randmin = -0.5, randmax = 0.5, randrange = randmax - randmin;
        double randval = randmin + (((double)rand()) / ((double)RAND_MAX / randrange));
        l.nodes[ni].weights.v[ii] = randval; // -0.5 <= randval <= 0.5
      }
      l.nodes[ni].output = 0.0;
      l.nodes[ni].delta = 0.0;
    }
    l.output.v = (double *)calloc(numnodes, sizeof(double));
    l.output.len = numnodes;
  }
  return l;
}

network initnetwork(int numinputs, int numoutputs) {
  network n;
  n.numlayers = 3;
  n.layers[0] = initlayer(numinputs, 1); // 1 node, len weights
  n.layers[1] = initlayer(numinputs, NETWORKNODES);
  n.layers[2] = initlayer(NETWORKNODES, numoutputs);
  n.output.v = (double *)calloc(numoutputs, sizeof(double));
  n.output.len = numoutputs;
  return n;
}

void setidealoutputindex(arr idealoutput, int idealindex) {
  int oi = -1;
  while (++oi < idealoutput.len)
    { idealoutput.v[oi] = (oi == idealindex) ? 1.0 : 0.0; }
}

void loadpattern(double *pattern, char *filename, int w, int h) {
  FILE *fp = fopen(filename, "r");
  if (!fp) { printf("openfile failed.\n"); return; }
  int rix = 0, cix = 0;
  while (1) {
    int inch = fgetc(fp);
    if (inch == EOF) { break; }
    if (rix >= h) { printf("too many rows.\n"); return; }
    if (inch == '*') {
//      if (output == 'y') { putchar('*'); }
      pattern[rix * w + cix] = 1.0;
      cix++;
    } else if (inch == ' ') {
//      if (output == 'y') { putchar(' '); }
      pattern[rix * w + cix] = 0.0;
      cix++;
    } else {
      if (cix == w && inch == '\n') {
//        if (output == 'y') { putchar('\n'); }
        ++rix;
        cix = 0;
      } else if (cix == w)
        { printf("linebreak expected."); return; }
      else { printf("unexpected char %d (cix=%d).", inch, cix); return; }
    }
  }
  fclose(fp);
}

void setnetworkinput(network *net, arr input) {
  int inputlen = net->layers[0].nodes[0].weights.len;
  if (inputlen != input.len) { printf("input mismatch.\n"); }
  int ii = -1;
  while (++ii < input.len)
    { net->layers[0].nodes[0].weights.v[ii] = input.v[ii]; }
}

double getlayeroutputvalue(network *net, int layerindex, int nodeindex) {
  if (layerindex == 0) { // input layer returns weights of node 0 (inputs)
    if (nodeindex >= net->layers[0].nodes[0].weights.len) { printf("oob0"); return 0.0; }
    return net->layers[0].nodes[0].weights.v[nodeindex];
  } else { // other layers return output weights of each of their nodes
//printf("{{%d %d}}", layerindex, nodeindex);
    if (nodeindex >= net->layers[layerindex].numnodes) { printf("oobN"); return 0.0; }
    return net->layers[layerindex].nodes[nodeindex].output;
  }
}

void copylayeroutput(network *net, int layerindex, arr output) {
  int outputlen = net->layers[layerindex].numnodes;
  if (layerindex == 0)
    { outputlen = net->layers[0].nodes[0].weights.len; }
  if (outputlen != output.len)
    { printf("mismatch %d %d", outputlen, output.len); return; }
  int lvi = -1;
  while (++lvi < output.len)
    { output.v[lvi] = getlayeroutputvalue(net, layerindex, lvi); }
} // used to copy node outputs to layer output buffer

void copynetworkoutput(network *net, arr target) {
//  layer *outlayer = &net->layers[n->numlayers - 1];
////  int ni = -1;
//  while (++ni < outlayer->numnodes && ni < target.len)
//    { target.v[ni] = outlayer->nodes[ni].output; }
  copylayeroutput(net, net->numlayers - 1, target);
} // network output is output of final layer

void feedpatternthroughnetwork(arr pattern, network *net) {
  setnetworkinput(net, pattern);
  copylayeroutput(net, 0, net->layers[0].output);
  int li = 0; // starts at layer 1
  while (++li < net->numlayers) { // 1, 2
    int ni = -1; // each node
    while (++ni < net->layers[li].numnodes) {
      node *np = &net->layers[li].nodes[ni];
      calculateoutputvalue(net->layers[li - 1].output, np);
      copylayeroutput(net, li, net->layers[li].output);
    }
  } // backpropagation uses output, only changes deltas/weights
  copynetworkoutput(net, net->output); // happens again after bp, should be unnecessary
}

void calculatedeltas(network *net, arr ideal) {
  node *outputnodes = net->layers[net->numlayers - 1].nodes;
  int numoutputnodes = net->layers[net->numlayers - 1].numnodes;
  if (numoutputnodes != ideal.len) { printf("delta mismatch.\n"); return; }
  int oni = -1; // first calculate for output layer nodes
  while (++oni < numoutputnodes && oni < ideal.len) {
    double outval = outputnodes[oni].output;
    outputnodes[oni].delta = outval * (1.0 - outval) * (ideal.v[oni] - outval);
  }
  int hli = net->numlayers - 1;
  while (--hli > 0) { // then nodes of preceding hidden layers
    node *hiddennodes = net->layers[hli].nodes;
    int numhiddennodes = net->layers[hli].numnodes;
    int hni = -1;
    while (++hni < numhiddennodes) {
      double laterstagesum = 0.0;
      node *laternodes = net->layers[hli + 1].nodes;
      int numlaternodes = net->layers[hli + 1].numnodes;
      int lni = -1;
      while (++lni < numlaternodes) {
        double weight = laternodes[lni].weights.v[hni + 1];
        laterstagesum += laternodes[lni].delta * weight;
      } // hni + 1 because weights[0] is a reference weight
      double outval = hiddennodes[hni].output;
      hiddennodes[hni].delta = outval * (1.0 - outval) * laterstagesum;
    }
  }
}

void updateweights(network *net) {
  int numlayers = net->numlayers;
  int uli = 0; // skip input layer
  while (++uli < numlayers) {
    node *layernodes = net->layers[uli].nodes;
    int numlayernodes = net->layers[uli].numnodes;
    int lni = -1; // each node
    while (++lni < numlayernodes) {
      double addweight = LEARNINGCOEFFICIENT * layernodes[lni].delta * -1.0;
      layernodes[lni].weights.v[0] += addweight; // bias weight
      int wi = 0; // skip weight 0 (bias weight), offset input -1
      while (++wi < layernodes[lni].weights.len) {
        double input = getlayeroutputvalue(net, uli - 1, wi - 1); //lni);
        addweight = LEARNINGCOEFFICIENT * layernodes[lni].delta * input;
        layernodes[lni].weights.v[wi] += addweight;
      }
    }
  }
}

void backpropagatenetwork(network *net, arr idealoutput) {
  calculatedeltas(net, idealoutput);
  updateweights(net);
}

double calculateerror(arr actual, arr ideal) {
  double error = 0.0;
  int ei = -1;
  while (++ei < actual.len && ei < ideal.len) {
    double adderror = pow(actual.v[ei] - ideal.v[ei], 2.0);
    error += adderror;
  }
  return error;
}

void showpatternfile(int patternnumber, int noise, int line) {
  char *files[] = { "Clock", "Cross", "Exclamation", "Face",
                    "GiveWay", "House", "Info", "Smile",
                    "Stand", "Stop", "Tick", "Walk" };
//  int numfiles = 12;
  int patternindex = patternnumber - 1;
  if (line == -1) { printf("%s", files[patternindex]); return; }
  char filename[100]; // line 0 = all lines, else line N no \n
  sprintf(filename, "%s.%d.txt", files[patternindex], noise);
  FILE *fp = fopen(filename, "r");
  if (!fp) { printf("showfile failed.\n"); return; }
  int row = 1; // first line is line 1
  while (1) {
    int inch = fgetc(fp);
    if (inch == EOF) { break; }
    if (line == 0 || row == line) {
      if (inch == '*') {
        putchar('*');
      } else if (inch == ' ') {
        putchar(' ');
      } else if (inch == '\n') {
        if (row == line) { break; }
        else { putchar('\n'); ++row; }
      } else {
        printf("unexpected char %d.\n", inch);
        return;
      }
    } else if (inch == '\n') { ++row; }
  }
  fclose(fp);
}

void showpattern(int patternnumber, int noise) {
  showpatternfile(patternnumber, noise, 0); // all lines
}

void showpatternline(int patternnumber, int noise, int line) {
  showpatternfile(patternnumber, noise, line); // this line no \n
}

void teachpattern(network *net, int patternnumber, int noise, arr *output) {
  char *files[] = { "Clock", "Cross", "Exclamation", "Face",
                    "GiveWay", "House", "Info", "Smile",
                    "Stand", "Stop", "Tick", "Walk" };
  int numfiles = 12;
  double patternv[] = { 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0 };
  int patternwidth = 12, patternheight = 12, patternlen = 144; // 12 x 12
  arr pattern = { patternv, patternlen };
  char mode = 'L'; // learn mode for positive pattern number
  int patternindex = patternnumber - 1;
  if (patternnumber < 0) {
    mode = 'T';    // test mode for negative pattern number
    patternindex = (0 - patternnumber) - 1;
  }
  char filename[100];
  sprintf(filename, "%s.%d.txt", files[patternindex], noise);
  loadpattern(pattern.v, filename, patternwidth, patternheight);
  if (mode == 'L') { // LEARN mode - teach
    double idealoutputv[] = { 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0 };
    arr idealoutput = { idealoutputv, numfiles };
    setidealoutputindex(idealoutput, patternindex);
    feedpatternthroughnetwork(pattern, net); // calculate outputs
    backpropagatenetwork(net, idealoutput);  // update deltas, weights
//    copynetworkoutput(net, net->output);
    double error = calculateerror(net->output, idealoutput);
    if (output->len == 1) { output->v[0] = error; }
    else { printf("mismatch output.len != 1"); return; }
//    arr errorarr = { &error, 1 };
//    return errorarr;    // returns array with 1 value, error
  } else if (mode == 'T') { // TEST mode - check
    feedpatternthroughnetwork(pattern, net);
//    copynetworkoutput(net, net->output);
    if (output->len == net->output.len)
      { output->v = net->output.v; }
    else { printf("mismatch output.len != %d", net->output.len); return; }
//    return net->output; // returns array of confidence values
  } else { return; } // return net->output; } // no change to network
}

void fastteachpatterns(network *net, int first, int last, int iterations, int noise, arr *output) {
  int numfiles = last - first + 1;
  double totalerror = 0.0;
  int iter = 0;
  while (++iter <= iterations) {
    int patternnum = first - 1;
    while (++patternnum <= last) {
      double errorvalue = 0.0;
      arr error = { &errorvalue, 1 };
      teachpattern(net, patternnum, noise, &error);
      if (iter == iterations)
        { totalerror += error.v[0]; }
    }
  }
  double avgerror = totalerror / ((double)numfiles) / sqrt(2.0);
  if (output->len == 1) { output->v[0] = avgerror; }
  else { printf("fast output.len != 1\n"); return; }
//  arr avgerrorarr = { &avgerror, 1 };
//  return avgerrorarr;
}

int main(int argc, char **argv) {
  int patternlen = 144;
  int numfiles = 12;
  network net = initnetwork(patternlen, numfiles);
  printf("ZXC for 0%%, 5%%, 10%% noise, LTSF to switch mode\n");
  char mode = 'L';
  int noisevalue = 0;
  printf("Starting in LEARN mode with random memory -0.5 - 0.5\n");
  int fastiterations = 100;
  printf("FASTLEARN will teach %d iterations of all files.\n", fastiterations);
  while (1) {
    int patternnumber = 0;
    char input[10];
    while (mode != 'F' && (patternnumber < 1 || patternnumber > 12)) {
      char *modename = (mode == 'L') ? "LEARN" : (mode == 'T') ? "TEST" : "SHOW";
      printf("%s pattern 1-12: ", modename);
      scanf(" %9s", input);
      if (input[0] >= 'a' && input[0] <= 'z') { input[0] += 'A' - 'a'; }
      if (input[0] == 'Z') { noisevalue =  0; printf("Noise set to 0%%\n");   }
      if (input[0] == 'X') { noisevalue =  5; printf("Noise set to 5%%\n");   }
      if (input[0] == 'C') { noisevalue = 10; printf("Noise set to 10%%\n");  }
      if (input[0] == 'L') {      mode = 'L'; printf("Now in LEARN mode\n"); }
      if (input[0] == 'T') {      mode = 'T'; printf("Now in TEST mode\n");  }
      if (input[0] == 'S') {      mode = 'S'; printf("Now in SHOW mode\n");  }
      if (input[0] == 'F') {      mode = 'F'; printf("FASTLEARN mode... learning...\n");  }
      if (input[0] >= '0' && input[0] <= '9') {
        if (input[1] >= '0' && input[1] <= '9') {
          patternnumber = ((input[0] - '0') * 10) + (input[1] - '0');
        } else { patternnumber = (input[0] - '0'); }
      } // up to 99 patterns
    }
    if (mode == 'S') {
      showpattern(patternnumber, noisevalue);
    } else if (mode == 'L') {
      showpattern(patternnumber, noisevalue);
      double errorvalue = 0.0;
      arr error = { &errorvalue, 1 };
      teachpattern(&net, patternnumber, noisevalue, &error);
      printf("Error after teach: %f\n", error.v[0]);
    } else if (mode == 'T') {
      arr confidence = { NULL, numfiles };
      teachpattern(&net, -patternnumber, noisevalue, &confidence);
      double maxconfidence = 0.0;
      int maxconfidenceix = -1;
      int ci = -1;
      while (++ci < confidence.len) {
        if (confidence.v[ci] > maxconfidence)
          { maxconfidence = confidence.v[ci]; maxconfidenceix = ci; }
      }
      ci = -1;
      while (++ci < confidence.len) {
        showpatternline(patternnumber, noisevalue, ci + 1);
        printf("    %2d: %5.1f%%  ", ci + 1, confidence.v[ci] * 100.0);
        if (ci == maxconfidenceix)
          { showpatternfile(ci + 1, noisevalue, -1); }
        printf("\n");
      }
    } else if (mode == 'F') {
      double errorvalue = 0.0;
      arr error = { &errorvalue, 1 };
      fastteachpatterns(&net, 1, 12, fastiterations, noisevalue, &error);
      printf("RMS error after %d iterations of %d files: %f\n",
               fastiterations, 12, error.v[0]);
      mode = 'T';
    }
  }
  return 0;
}
