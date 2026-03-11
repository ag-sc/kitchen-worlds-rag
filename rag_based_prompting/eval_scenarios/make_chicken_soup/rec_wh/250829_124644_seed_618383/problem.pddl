
(define
  (problem test_kitchen_chicken_soup_250829_124644_seed_618383)
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

	(canpull left)
	(canpull right)

	(cangrasphandle)
	(handempty left)
	(handempty right)

	(food chicken-leg)

	(controllable right)
	(controllable left)

	(space braiserbody#1)
	(space counter#1::sektion)

	(sprinkler salt-shaker)
	(sprinkler pepper-shaker)

	(graspable pepper-shaker)
	(graspable chicken-leg)
	(graspable salt-shaker)
	(graspable braiserbody#1)
	(graspable braiserlid#1)

	(bconf q968=(2.0, 6.25, 0.2, 3.142))

	(region counter#1::hitman_countertop)
	(region fridge#1::shelf_top)
	(region counter#1::front_left_stove)
	(region counter#1::sektion)
	(region counter#1::front_right_stove)
	(region basin#1::basin_bottom)
	(region braiserbody#1)
	(region braiserbody#1::braiser_bottom)
	(region counter#1::indigo_tmp)

	(atbconf q968=(2.0, 6.25, 0.2, 3.142))
	(surface counter#1::front_right_stove)
	(surface basin#1::basin_bottom)
	(surface braiserbody#1::braiser_bottom)
	(surface braiserbody#1)
	(surface counter#1::hitman_countertop)
	(surface fridge#1::shelf_top)
	(surface counter#1::front_left_stove)
	(surface counter#1::indigo_tmp)

	(containable chicken-leg braiserbody#1)
	(containable salt-shaker counter#1::sektion)
	(containable braiserlid#1 counter#1::sektion)
	(containable chicken-leg counter#1::sektion)
	(containable braiserbody#1 counter#1::sektion)
	(containable pepper-shaker counter#1::sektion)

	(door counter#1::chewie_door_left_joint)
	(door counter#1::chewie_door_right_joint)
	(door fridge#1::fridge_door)

	(staticlink counter#1::front_right_stove)
	(staticlink basin#1::basin_bottom)
	(staticlink braiserbody#1)
	(staticlink braiserbody#1::braiser_bottom)
	(staticlink counter#1::indigo_tmp)
	(staticlink counter#1::hitman_countertop)
	(staticlink fridge#1::shelf_top)
	(staticlink counter#1::front_left_stove)
	(staticlink counter#1::sektion)

	(isjointto fridge#1::fridge_door fridge#1)
	(isjointto faucet#1::joint_faucet_0 faucet#1)
	(isjointto oven#1::knob_joint_3 oven#1)
	(isjointto counter#1::chewie_door_left_joint counter#1)
	(isjointto counter#1::chewie_door_right_joint counter#1)
	(isjointto oven#1::knob_joint_2 oven#1)
	(joint counter#1::chewie_door_right_joint)
	(joint faucet#1::joint_faucet_0)
	(joint fridge#1::fridge_door)
	(joint oven#1::knob_joint_3)
	(joint oven#1::knob_joint_2)
	(joint counter#1::chewie_door_left_joint)

	(position fridge#1::fridge_door pstn1209=1.78)
	(position oven#1::knob_joint_3 pstn1211=0.0)
	(position oven#1::knob_joint_2 pstn1210=0.0)
	(position counter#1::chewie_door_left_joint pstn1208=-1.872)
	(position counter#1::chewie_door_right_joint pstn1207=1.872)
	(position faucet#1::joint_faucet_0 pstn1212=0.0)

	(isclosedposition oven#1::knob_joint_3 pstn1211=0.0)
	(isclosedposition oven#1::knob_joint_2 pstn1210=0.0)
	(isclosedposition faucet#1::joint_faucet_0 pstn1212=0.0)
	(unattachedjoint counter#1::chewie_door_right_joint)
	(unattachedjoint fridge#1::fridge_door)
	(unattachedjoint oven#1::knob_joint_3)
	(unattachedjoint faucet#1::joint_faucet_0)
	(unattachedjoint oven#1::knob_joint_2)
	(unattachedjoint counter#1::chewie_door_left_joint)

	(isopenedposition fridge#1::fridge_door pstn1209=1.78)
	(isopenedposition counter#1::chewie_door_left_joint pstn1208=-1.872)
	(isopenedposition counter#1::chewie_door_right_joint pstn1207=1.872)
	(stackable pepper-shaker counter#1::hitman_countertop)
	(stackable braiserbody#1 counter#1::indigo_tmp)
	(stackable chicken-leg counter#1::indigo_tmp)
	(stackable salt-shaker counter#1::indigo_tmp)
	(stackable braiserlid#1 braiserbody#1)
	(stackable salt-shaker counter#1::hitman_countertop)
	(stackable braiserlid#1 counter#1::indigo_tmp)
	(stackable braiserlid#1 counter#1::hitman_countertop)
	(stackable chicken-leg basin#1::basin_bottom)
	(stackable braiserlid#1 counter#1::front_left_stove)
	(stackable pepper-shaker counter#1::indigo_tmp)
	(stackable braiserlid#1 basin#1::basin_bottom)
	(stackable chicken-leg braiserbody#1::braiser_bottom)

	(atposition counter#1::chewie_door_left_joint pstn1208=-1.872)
	(atposition faucet#1::joint_faucet_0 pstn1212=0.0)
	(atposition counter#1::chewie_door_right_joint pstn1207=1.872)
	(atposition oven#1::knob_joint_3 pstn1211=0.0)
	(atposition oven#1::knob_joint_2 pstn1210=0.0)
	(atposition fridge#1::fridge_door pstn1209=1.78)

	(atpose braiserbody#1 p3307=(0.7, 8.9, 0.923, 0.0, -0.0, 1.571))
	(atpose salt-shaker p3311=(0.771, 7.071, 1.152, 0.0, -0.0, 3.142))
	(atpose chicken-leg p3308=(0.654, 4.846, 1.384, 0.0, 0.0, -0.366))
	(atpose pepper-shaker p3312=(0.764, 7.303, 1.164, 0.0, -0.0, 3.142))
	(atpose braiserlid#1 p3309=(0.567, 7.872, 0.712, 0.0, -0.0, 2.682))
	(pose chicken-leg p3308=(0.654, 4.846, 1.384, 0.0, 0.0, -0.366))
	(pose pepper-shaker p3312=(0.764, 7.303, 1.164, 0.0, -0.0, 3.142))
	(pose braiserbody#1 p3307=(0.7, 8.9, 0.923, 0.0, -0.0, 1.571))
	(pose braiserlid#1 p3309=(0.567, 7.872, 0.712, 0.0, -0.0, 2.682))
	(pose salt-shaker p3311=(0.771, 7.071, 1.152, 0.0, -0.0, 3.142))

	(aconf right aq648=(-0.677, -0.343, -1.2, -1.467, -1.242, -1.954, -2.223))
	(aconf left aq632=(0.677, -0.343, 1.2, -1.467, 1.242, -1.954, 2.223))

	(ataconf right aq648=(-0.677, -0.343, -1.2, -1.467, -1.242, -1.954, -2.223))
	(ataconf left aq632=(0.677, -0.343, 1.2, -1.467, 1.242, -1.954, 2.223))

	(contained pepper-shaker p3312=(0.764, 7.303, 1.164, 0.0, -0.0, 3.142) counter#1::sektion)
	(contained salt-shaker p3311=(0.771, 7.071, 1.152, 0.0, -0.0, 3.142) counter#1::sektion)

	(supported braiserlid#1 p3309=(0.567, 7.872, 0.712, 0.0, -0.0, 2.682) counter#1::front_left_stove)
	(supported braiserbody#1 p3310=(0.7, 8.9, 0.923, 0.0, -0.0, 1.571) counter#1::indigo_tmp)

  )

  (:goal (and
    (open fridge#1::fridge_door)
  ))
)
        