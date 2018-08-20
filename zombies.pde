int zombieCount = 50; // high numbers can take a lot of training time
int unitsize = 15;
int cellsize = 80;
int rounds = 0;
int iters = 50000;
int roundLimit = 100000;
int shootcooldown = 0;
float speed = 4;
double best = Double.MIN_VALUE;

int cells = 4;
int inputCount = cells*cells + 4;
int hiddenCount = 4;

double input[] = new double[inputCount];

double weights1[] = new double[inputCount*hiddenCount];
double weights2[] = new double[hiddenCount];
double bias[] = new double[hiddenCount];

double weights1Best[] = new double[inputCount*hiddenCount];
double weights2Best[] = new double[hiddenCount];
double biasBest[] = new double[hiddenCount];

ArrayList<Zombie> zombies = new ArrayList<Zombie>();
ArrayList<Bullet> bullets = new ArrayList<Bullet>();
Player player;

boolean gameOver = false;
boolean canShoot = true;
boolean training = false;

/////////////////////////////////////////////////////
// Classes
class Unit {
  public float x, y; 
  public Unit(float x, float y) {
    this.x = x;
    this.y = y;
  }

  public float dist(Unit u) {
    return sqrt(pow(x - u.x, 2) + pow(y - u.y, 2));
  }
}

class Zombie extends Unit {
  public Zombie(float x, float y) {
    super(x, y);
  }
}

class Player extends Unit {
  public Player(float x, float y) {
    super(x, y);
  }
}

class Bullet extends Unit {
  public Bullet(float x, float y) {
    super(x, y);
  }
  float targetx, targety;
}
//////////////////////////////////////////////////////

void setup() {
  fullScreen();
  width = 1400;
  init();
  randomize();
}

void init() {
  gameOver = false;
  rounds = 0;
  shootcooldown = 0;
  randomSeed(7);
  player = new Player(width / 2, height / 2);

  zombies.clear();
  for (int i = 0; i < zombieCount; i++) {
    Zombie m = null;
    while (m == null || m.dist(player) < 500) {
      m = new Zombie(random(width), random(height));
    }
    zombies.add(m);
  }
  randomSeed(System.currentTimeMillis());
}

float getRandom() {
  float range = 1;
  return random(2*range) - range;
}

void randomize() {
  // Init weights1
  for (int i = 0; i < weights1.length; i++) {
    weights1[i] = getRandom();
    weights1Best[i] = weights1[i];
  }
  
  // Init weights2
  weights2[0] =  0.07;
  weights2Best[0] = weights2[0];
  weights2[1] = 0.17;
  weights2Best[1] = weights2[1];
  weights2[2] = 0.71;
  weights2Best[2] = weights2[2];
  weights2[3] = 0.91;
  weights2Best[3] = weights2[3];

  // Init bias
  for (int i = 0; i < bias.length; i++) {
    bias[i] = getRandom();
    biasBest[i] = bias[i];
  }
}

float compute() {
  // Convert input to hidden layer
  double hidden[] = new double[hiddenCount];

  for (int i = 0; i < input.length; i++) {
    for (int j = 0; j < hidden.length; j++) {
      hidden[j] += input[i] * weights1[i * hiddenCount + j];
    }
  }

  //Apply activator
  for (int j = 0; j < hidden.length; j++) {
    hidden[j] = Math.tanh(hidden[j] + bias[j]);
  }

  // Convert hidden layer to output
  float output = 0;
  for (int j = 0; j < hidden.length; j++) {
    output += weights2[j] * hidden[j];
  }
  return output;
}

void shoot() {
  if (shootcooldown > 0)
    shootcooldown--;

  //Move bullets
  for (Bullet b : bullets) {
    Unit u = new Unit(b.targetx, b.targety);
    float d = b.dist(u);
    b.x += b.targetx / d * 10;
    b.y += b.targety / d * 10;
  }

  //Remove dead n bullets
  for (int j = 0; j < bullets.size(); j++) {

    // Out of screen
    if (bullets.get(j).x < -100 ||
      bullets.get(j).y < -100 ||
      bullets.get(j).x > width+100 ||
      bullets.get(j).y > height+100) {

      bullets.remove(j);
      j--;
    }
  }

  // Targets shot
  for (int i = 0; i < zombies.size(); i++) {
    for (int j = 0; j < bullets.size(); j++) {
      Zombie m = zombies.get(i);
      Bullet b = bullets.get(j);
      if (m.dist(b) < unitsize) {

        //Remove bullet
        bullets.remove(j);
        j--;

        //Remove zombie
        zombies.remove(i);
        i--;

        break;
      }
    }
  }

  // Check if we can shoot
  if (shootcooldown > 0 || zombies.size() == 0) return;

  int j = 0;
  for (int i = 0; i < zombies.size(); i++) {
    if (player.dist(zombies.get(i)) < player.dist(zombies.get(j))) {
      j = i;
    }
  }

  Bullet b = new Bullet(player.x, player.y);
  b.targetx = (zombies.get(j).x - player.x) * 1000;
  b.targety = (zombies.get(j).y - player.y) * 1000;

  bullets.add(b);
  shootcooldown = 10;
}

void update() {
  // Reset input
  for (int i = 0; i < input.length; i++) {
    input[i] = 0;
  }

  rounds++;

  // Zombie move
  for (int i = 0; i < zombies.size(); i++) {
    float d = player.dist(zombies.get(i));
    float x = player.x - zombies.get(i).x;
    float y = player.y - zombies.get(i).y;

    if (d >= 1) {
      zombies.get(i).x += (x / d * min(speed, d));
      zombies.get(i).y += (y / d * min(speed, d));
    }
  }

  //Detect zombies
  for (int i = 0; i < zombies.size(); i++) {
    Zombie m = zombies.get(i);

    //Check 16 cells
    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 4; k++) {

        // Check x
        if (  m.x > (player.x - 2*cellsize) + (j * cellsize) && m.x < (player.x - 2*cellsize) + (j+1) * cellsize) {
          // Check y
          if (m.y > (player.y - 2*cellsize) + (k * cellsize) && m.y < (player.y - 2*cellsize) + (k+1) * cellsize) {
            input[(j * 4 + k)] = 1;
          }
        }
      }
    }
  }

  // Detect walls
  int bound3 = 200;
  if (player.x < bound3) {
    input[inputCount - 4] = 1;
  }
  if (player.x > width-bound3) {
    input[inputCount - 3] = 1;
  }
  if (player.y < bound3) {
    input[inputCount - 2] = 1;
  }
  if (player.y > height-bound3) {
    input[inputCount - 1] = 1;
  }

  // Apply move
  float res = compute();
  player.x += 1.5*speed*cos(res * 2*PI);
  player.y += 1.5*speed*sin(res * 2*PI);

  // Check if out of bounds
  int bound = unitsize;
  if (player.x < bound) {
    player.x = bound;
  }
  if (player.x > width-bound) {
    player.x = width - bound;
  }
  if (player.y < bound) {
    player.y = bound;
  }
  if (player.y > height-bound) {
    player.y = height-bound;
  }

  // Check for gameOver
  for (int i = 0; i < zombies.size(); i++) {
    if (player.dist(zombies.get(i)) <= unitsize) {
      gameOver = true; 
      break;
    }
  }

  if (training && rounds > roundLimit) {
    gameOver = true;
  }

  if (canShoot)
    shoot();
}


void train() {
  training = true;
  for (int j = 0; j < iters; j++) {
    if (best > roundLimit) break;

    init();
    while (!gameOver) {
      update();

      if (gameOver) {
        if (rounds > best) {
          best = rounds; 
          // Copy current weights into best
          for (int i = 0; i < weights1.length; i++) {
            weights1Best[i] = weights1[i];
          }
          for (int i = 0; i < weights2.length; i++) {
            weights2Best[i] =  weights2[i];
          }
          for (int i = 0; i < bias.length; i++) {
            biasBest[i] = bias[i];
          }
        }
        reset();

        for (int i = 0; i < weights1.length; i++) {
          if (input[i / hiddenCount] == 1) {
            weights1[i] += 100*getRandom();
          }
        }
      }
    }
  }

  //Assign bestweights to weights
  reset();
  training = false;
}

void reset() {
  for (int i = 0; i < weights1.length; i++) {
    weights1[i] = weights1Best[i];
  }
  for (int i = 0; i < weights2.length; i++) {
    weights2[i] = weights2Best[i];
  }
  for (int i = 0; i < bias.length; i++) {
    bias[i] =  biasBest[i];
  }
}


int toggle = 0;
int toggleBound = 180;

int waittime = 600;
int generations = 1;


void draw() {
  strokeWeight(2);
  background(color(0, 150, 200));

  // Victory or defeat
  if (gameOver) {
    //Victory
    if (toggle > toggleBound) {
      toggle = 0;
      best = Double.MIN_VALUE;
    }
    //Defeat
    else {
      generations++;
      delay(1000);
      randomize();
      train();
    }
    init();
  }

  //Raster
  fill(color(255, 255, 255, 0));
  strokeWeight(1);
  for (int j = 0; j < 4; j++) {
    for (int k = 0; k < 4; k++) {
      rect(player.x + j * cellsize - 2* cellsize, player.y + k * cellsize - 2* cellsize, cellsize, cellsize);
    }
  }
  strokeWeight(2);

  // Dots
  fill(color(0, 255, 0));
  ellipse(player.x, player.y, unitsize, unitsize);

  fill(color(255, 0, 0));
  for (int i = 0; i < zombies.size(); i++) {
    ellipse(zombies.get(i).x, zombies.get(i).y, unitsize, unitsize);
  }

  //Bullets
  for (int i = 0; i < bullets.size(); i++) {
    fill(color(255, 255, 0));
    ellipse(bullets.get(i).x, bullets.get(i).y, unitsize, unitsize);
  }
  
  //Right plane
  fill(80);
  rect(1400, -10, 1000, 1200);
  
  //Stats
  textSize(64);
  textAlign(CENTER, CENTER);
  fill(color(255, 255, 255));

  int base = width + 260;
  text("Generation: " + generations, base, 100);
  text("zombies: " + zombies.size(), base, 200);
  if (!canShoot)
    text("Time left: " + max(0, (waittime - rounds) / 60), base, 300);

  textSize(32);
  textAlign(CENTER, CENTER);
  fill(color(255, 255, 255, 128));
  text("Subscribe Daporan!", base, height - 50);

  //Victory
  textSize(128);
  textAlign(CENTER, CENTER);
  fill(color(255, 255, 255));
  if (zombies.size() == 0 || rounds >= waittime && !canShoot)
    text("Victory!", width / 2, height / 2);
  
  // Update
  for (int k = 0; k < 1; k++) {
    update();
    if (gameOver) {
      fill(color(255, 255, 255));
      text("Defeat!", width/2, height/2); 
      break;
    }
    //Victory
    if (zombies.size() == 0 || rounds >= waittime && !canShoot) {
      toggle++;
      if (toggle > toggleBound) {
        gameOver = true;
      }
    }
  }
}
