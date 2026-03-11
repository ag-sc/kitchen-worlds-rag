
(define
  (problem test_kitchen_chicken_soup_250828_114243_seed_426760)
  (:domain domain)

  (:objects
	base
	base-torso
	basin#1
	basin#1::basin_bottom
	braiserbody#1
	braiserbody#1::braiser_bottom
	braiserlid#1
	chicken-leg
	counter#1
	counter#1::chewie_door_left_joint
	counter#1::chewie_door_right_joint
	counter#1::front_left_stove
	counter#1::front_right_stove
	counter#1::hitman_countertop
	counter#1::indigo_tmp
	counter#1::sektion
	faucet#1
	faucet#1::joint_faucet_0
	fridge#1
	fridge#1::fridge_door
	fridge#1::shelf_top
	head
	left_arm
	left_gripper
	oven#1
	oven#1::knob_joint_2
	oven#1::knob_joint_3
	pepper-shaker
	right_arm
	right_gripper
	salt-shaker
	torso
  )

  (:init
	;; discrete facts (e.g. types, affordances)
	(canmove)
	(canpick)

	(arm right)
	(arm left)

	(canmovebase)

	(canpull right)
	(canpull left)

	(cangrasphandle)

	(handempty right)
	(handempty left)

	(food chicken-leg)

	(controllable right)
	(controllable left)

	(space braiserbody#1)
	(space counter#1::sektion)

	(graspable salt-shaker)
	(graspable braiserbody#1)
	(graspable braiserlid#1)
	(graspable pepper-shaker)
	(graspable chicken-leg)
	(sprinkler salt-shaker)
	(sprinkler pepper-shaker)

	(joint oven#1::knob_joint_3)
	(joint oven#1::knob_joint_2)
	(joint counter#1::chewie_door_left_joint)
	(joint counter#1::chewie_door_right_joint)
	(joint faucet#1::joint_faucet_0)
	(joint fridge#1::fridge_door)

	(surface fridge#1::shelf_top)
	(surface counter#1::hitman_countertop)
	(surface counter#1::front_left_stove)
	(surface braiserbody#1::braiser_bottom)
	(surface braiserbody#1)
	(surface counter#1::indigo_tmp)
	(surface basin#1::basin_bottom)
	(surface counter#1::front_right_stove)

	(region counter#1::indigo_tmp)
	(region basin#1::basin_bottom)
	(region counter#1::front_right_stove)
	(region counter#1::sektion)
	(region fridge#1::shelf_top)
	(region counter#1::hitman_countertop)
	(region counter#1::front_left_stove)
	(region braiserbody#1::braiser_bottom)
	(region braiserbody#1)

	(staticlink counter#1::indigo_tmp)
	(staticlink counter#1::front_right_stove)
	(staticlink fridge#1::shelf_top)
	(staticlink counter#1::sektion)
	(staticlink braiserbody#1::braiser_bottom)
	(staticlink counter#1::hitman_countertop)
	(staticlink counter#1::front_left_stove)
	(staticlink basin#1::basin_bottom)
	(staticlink braiserbody#1)

	(bconf q560=(2.0, 6.25, 0.2, 3.142))

	(atbconf q560=(2.0, 6.25, 0.2, 3.142))

	(isjointto oven#1::knob_joint_2 oven#1)
	(isjointto counter#1::chewie_door_left_joint counter#1)
	(isjointto counter#1::chewie_door_right_joint counter#1)
	(isjointto faucet#1::joint_faucet_0 faucet#1)
	(isjointto oven#1::knob_joint_3 oven#1)
	(isjointto fridge#1::fridge_door fridge#1)

	(door counter#1::chewie_door_right_joint)
	(door fridge#1::fridge_door)
	(door counter#1::chewie_door_left_joint)

	(containable braiserbody#1 counter#1::sektion)
	(containable chicken-leg counter#1::sektion)
	(containable salt-shaker counter#1::sektion)
	(containable braiserlid#1 counter#1::sektion)
	(containable pepper-shaker counter#1::sektion)
	(containable chicken-leg braiserbody#1)

	(atposition faucet#1::joint_faucet_0 pstn5=0.0)
	(atposition counter#1::chewie_door_left_joint pstn1=-1.872)
	(atposition counter#1::chewie_door_right_joint pstn0=1.872)
	(atposition oven#1::knob_joint_3 pstn4=0.0)
	(atposition fridge#1::fridge_door pstn2=1.78)
	(atposition oven#1::knob_joint_2 pstn3=0.0)
	(stackable braiserbody#1 counter#1::indigo_tmp)
	(stackable braiserlid#1 counter#1::hitman_countertop)
	(stackable chicken-leg counter#1::indigo_tmp)
	(stackable salt-shaker counter#1::indigo_tmp)
	(stackable braiserlid#1 counter#1::front_left_stove)
	(stackable braiserlid#1 basin#1::basin_bottom)
	(stackable braiserlid#1 braiserbody#1)
	(stackable pepper-shaker counter#1::hitman_countertop)
	(stackable braiserlid#1 counter#1::indigo_tmp)
	(stackable chicken-leg braiserbody#1::braiser_bottom)
	(stackable salt-shaker counter#1::hitman_countertop)
	(stackable pepper-shaker counter#1::indigo_tmp)
	(stackable chicken-leg basin#1::basin_bottom)

	(isclosedposition oven#1::knob_joint_2 pstn3=0.0)
	(isclosedposition faucet#1::joint_faucet_0 pstn5=0.0)
	(isclosedposition oven#1::knob_joint_3 pstn4=0.0)

	(unattachedjoint counter#1::chewie_door_right_joint)
	(unattachedjoint fridge#1::fridge_door)
	(unattachedjoint oven#1::knob_joint_3)
	(unattachedjoint faucet#1::joint_faucet_0)
	(unattachedjoint oven#1::knob_joint_2)
	(unattachedjoint counter#1::chewie_door_left_joint)

	(position counter#1::chewie_door_left_joint pstn1=-1.872)
	(position faucet#1::joint_faucet_0 pstn5=0.0)
	(position counter#1::chewie_door_right_joint pstn0=1.872)
	(position fridge#1::fridge_door pstn2=1.78)
	(position oven#1::knob_joint_3 pstn4=0.0)
	(position oven#1::knob_joint_2 pstn3=0.0)

	(pose braiserlid#1 p2=(0.567, 7.872, 0.712, 0.0, -0.0, 1.352))
	(pose chicken-leg p1=(0.654, 4.846, 1.384, 0.0, 0.0, -0.366))
	(pose salt-shaker p4=(0.771, 7.071, 1.152, 0.0, -0.0, 3.142))
	(pose pepper-shaker p5=(0.764, 7.303, 1.164, 0.0, -0.0, 3.142))
	(pose braiserbody#1 p0=(0.7, 8.954, 0.923, 0.0, -0.0, 1.571))

	(atpose braiserbody#1 p0=(0.7, 8.954, 0.923, 0.0, -0.0, 1.571))
	(atpose braiserlid#1 p2=(0.567, 7.872, 0.712, 0.0, -0.0, 1.352))
	(atpose salt-shaker p4=(0.771, 7.071, 1.152, 0.0, -0.0, 3.142))
	(atpose chicken-leg p1=(0.654, 4.846, 1.384, 0.0, 0.0, -0.366))
	(atpose pepper-shaker p5=(0.764, 7.303, 1.164, 0.0, -0.0, 3.142))

	(isopenedposition counter#1::chewie_door_left_joint pstn1=-1.872)
	(isopenedposition fridge#1::fridge_door pstn2=1.78)
	(isopenedposition counter#1::chewie_door_right_joint pstn0=1.872)

	(aconf left aq704=(0.677, -0.343, 1.2, -1.467, 1.242, -1.954, 2.223))
	(aconf right aq608=(-0.677, -0.343, -1.2, -1.467, -1.242, -1.954, -2.223))

	(ataconf left aq704=(0.677, -0.343, 1.2, -1.467, 1.242, -1.954, 2.223))
	(ataconf right aq608=(-0.677, -0.343, -1.2, -1.467, -1.242, -1.954, -2.223))

	(contained pepper-shaker p5=(0.764, 7.303, 1.164, 0.0, -0.0, 3.142) counter#1::sektion)
	(contained salt-shaker p4=(0.771, 7.071, 1.152, 0.0, -0.0, 3.142) counter#1::sektion)

	(supported braiserbody#1 p3=(0.7, 8.954, 0.923, 0.0, -0.0, 1.571) counter#1::indigo_tmp)
	(supported braiserlid#1 p2=(0.567, 7.872, 0.712, 0.0, -0.0, 1.352) counter#1::front_left_stove)

  )

  (:goal (and
    (open fridge#1::fridge_door)
  ))
)
        