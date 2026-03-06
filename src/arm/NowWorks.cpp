#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pca9685 = Adafruit_PWMServoDriver(0x40);

#define PWM_FREQ 50
#define NUM_SERVOS 5

const uint8_t servoChannels[NUM_SERVOS] = {0,1,2,3,4};

#define SERVO_MIN_US 500
#define SERVO_MAX_US 2500

#define STATE_DELAY 2000   // 2 seconds

// =====================
// FSM STATES
// =====================
enum RobotState {
  INIT,
  MOVE_TO_OBJECT,
  CLOSE_GRIPPER,
  MOVE_TO_USER,
  OPEN_GRIPPER,
  DONE
};

RobotState currentState = INIT;

unsigned long stateStart = 0;
bool stateExecuted = false;

// =====================
// SERVO CONVERSION
// =====================
uint16_t usToTicks(uint16_t us){
  return (uint16_t)((us / 20000.0) * 4096.0);
}

uint16_t angleToTicks(uint8_t angle){
  angle = constrain(angle,0,180);
  uint16_t us = map(angle,0,180,SERVO_MIN_US,SERVO_MAX_US);
  return usToTicks(us);
}

void setServo(uint8_t servo, uint8_t angle){
  pca9685.setPWM(servoChannels[servo],0,angleToTicks(angle));

  Serial.print("Servo ");
  Serial.print(servo);
  Serial.print(" -> ");
  Serial.println(angle);
}

// =====================
// STATE FUNCTIONS
// =====================

void runINIT(){
  Serial.println("INIT");

  setServo(0,40);
  setServo(1,20);
  setServo(2,95);
  setServo(3,80);
  setServo(4,30);   // gripper open
}

void runMove(){
  Serial.println("MOVE TO OBJECT");
  setServo(0,0);
}

void runClose(){
  Serial.println("CLOSE GRIPPER");
  setServo(4,0);   // close
}

void runMoveUser(){
  Serial.println("MOVE TO USER");
  setServo(0,90);
}

void runOpen(){
  Serial.println("OPEN GRIPPER");
  setServo(4,30);  // open again
}

// =====================
// SETUP
// =====================
void setup(){

  Serial.begin(9600);

  Wire.begin();
  pca9685.begin();
  pca9685.setPWMFreq(PWM_FREQ);

  delay(50);

  stateStart = millis();
}

// =====================
// LOOP
// =====================
void loop(){

  if(!stateExecuted){

    switch(currentState){

      case INIT:
        runINIT();
        break;

      case MOVE_TO_OBJECT:
        runMove();
        break;

      case CLOSE_GRIPPER:
        runClose();
        break;

      case MOVE_TO_USER:
        runMoveUser();
        break;

      case OPEN_GRIPPER:
        runOpen();
        break;

      case DONE:
        Serial.println("DONE");
        while(true);
    }

    stateExecuted = true;
    stateStart = millis();
  }

  if(millis() - stateStart >= STATE_DELAY){

    stateExecuted = false;

    switch(currentState){
      case INIT: currentState = MOVE_TO_OBJECT; break;
      case MOVE_TO_OBJECT: currentState = CLOSE_GRIPPER; break;
      case CLOSE_GRIPPER: currentState = MOVE_TO_USER; break;
      case MOVE_TO_USER: currentState = OPEN_GRIPPER; break;
      case OPEN_GRIPPER: currentState = DONE; break;
      default: break;
    }
  }
}
