
(define
  (problem test_kitchen_chicken_soup_250826_183436_seed_854983)
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

	(arm left)
	(arm right)

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

	(graspable chicken-leg)
	(graspable salt-shaker)
	(graspable braiserbody#1)
	(graspable braiserlid#1)
	(graspable pepper-shaker)
	(surface braiserbody#1)
	(surface counter#1::indigo_tmp)
	(surface basin#1::basin_bottom)
	(surface counter#1::front_right_stove)
	(surface counter#1::hitman_countertop)
	(surface fridge#1::shelf_top)
	(surface braiserbody#1::braiser_bottom)
	(surface counter#1::front_left_stove)

	(sprinkler pepper-shaker)
	(sprinkler salt-shaker)

	(region counter#1::sektion)
	(region braiserbody#1)
	(region braiserbody#1::braiser_bottom)
	(region counter#1::front_left_stove)
	(region counter#1::indigo_tmp)
	(region basin#1::basin_bottom)
	(region counter#1::front_right_stove)
	(region counter#1::hitman_countertop)
	(region fridge#1::shelf_top)

	(joint faucet#1::joint_faucet_0)
	(joint fridge#1::fridge_door)
	(joint oven#1::knob_joint_2)
	(joint oven#1::knob_joint_3)
	(joint counter#1::chewie_door_left_joint)
	(joint counter#1::chewie_door_right_joint)

	(bconf q960=(2.0, 6.25, 0.2, 3.142))

	(atbconf q960=(2.0, 6.25, 0.2, 3.142))

	(door counter#1::chewie_door_left_joint)
	(door counter#1::chewie_door_right_joint)
	(door fridge#1::fridge_door)
	(staticlink counter#1::front_left_stove)
	(staticlink basin#1::basin_bottom)
	(staticlink braiserbody#1)
	(staticlink counter#1::indigo_tmp)
	(staticlink counter#1::front_right_stove)
	(staticlink counter#1::hitman_countertop)
	(staticlink fridge#1::shelf_top)
	(staticlink braiserbody#1::braiser_bottom)
	(staticlink counter#1::sektion)

	(isjointto fridge#1::fridge_door fridge#1)
	(isjointto counter#1::chewie_door_right_joint counter#1)
	(isjointto oven#1::knob_joint_2 oven#1)
	(isjointto counter#1::chewie_door_left_joint counter#1)
	(isjointto faucet#1::joint_faucet_0 faucet#1)
	(isjointto oven#1::knob_joint_3 oven#1)

	(stackable chicken-leg basin#1::basin_bottom)
	(stackable braiserbody#1 counter#1::indigo_tmp)
	(stackable salt-shaker counter#1::indigo_tmp)
	(stackable chicken-leg counter#1::indigo_tmp)
	(stackable salt-shaker counter#1::hitman_countertop)
	(stackable braiserlid#1 counter#1::front_left_stove)
	(stackable braiserlid#1 counter#1::indigo_tmp)
	(stackable braiserlid#1 basin#1::basin_bottom)
	(stackable chicken-leg braiserbody#1::braiser_bottom)
	(stackable braiserlid#1 counter#1::hitman_countertop)
	(stackable braiserlid#1 braiserbody#1)
	(stackable pepper-shaker counter#1::indigo_tmp)
	(stackable pepper-shaker counter#1::hitman_countertop)

	(containable braiserbody#1 counter#1::sektion)
	(containable chicken-leg counter#1::sektion)
	(containable salt-shaker counter#1::sektion)
	(containable chicken-leg braiserbody#1)
	(containable braiserlid#1 counter#1::sektion)
	(containable pepper-shaker counter#1::sektion)

	(unattachedjoint counter#1::chewie_door_left_joint)
	(unattachedjoint counter#1::chewie_door_right_joint)
	(unattachedjoint faucet#1::joint_faucet_0)
	(unattachedjoint fridge#1::fridge_door)
	(unattachedjoint oven#1::knob_joint_3)
	(unattachedjoint oven#1::knob_joint_2)

	(isclosedposition oven#1::knob_joint_3 pstn7443=0.0)
	(isclosedposition faucet#1::joint_faucet_0 pstn7444=0.0)
	(isclosedposition oven#1::knob_joint_2 pstn7442=0.0)

	(position counter#1::chewie_door_right_joint pstn7439=1.872)
	(position fridge#1::fridge_door pstn7441=1.78)
	(position faucet#1::joint_faucet_0 pstn7444=0.0)
	(position oven#1::knob_joint_3 pstn7443=0.0)
	(position oven#1::knob_joint_2 pstn7442=0.0)
	(position counter#1::chewie_door_left_joint pstn7440=-1.872)

	(atposition counter#1::chewie_door_left_joint pstn7440=-1.872)
	(atposition fridge#1::fridge_door pstn7441=1.78)
	(atposition oven#1::knob_joint_2 pstn7442=0.0)
	(atposition counter#1::chewie_door_right_joint pstn7439=1.872)
	(atposition oven#1::knob_joint_3 pstn7443=0.0)
	(atposition faucet#1::joint_faucet_0 pstn7444=0.0)

	(pose salt-shaker p10295=(0.771, 7.071, 1.152, 0.0, -0.0, 3.142))
	(pose braiserbody#1 p10291=(0.7, 8.9, 0.923, 0.0, -0.0, 1.571))
	(pose pepper-shaker p10296=(0.764, 7.303, 1.164, 0.0, -0.0, 3.142))
	(pose chicken-leg p10292=(0.654, 4.846, 1.384, 0.0, 0.0, -0.366))
	(pose braiserlid#1 p10293=(0.567, 7.872, 0.712, 0.0, -0.0, 1.303))

	(atpose salt-shaker p10295=(0.771, 7.071, 1.152, 0.0, -0.0, 3.142))
	(atpose braiserbody#1 p10291=(0.7, 8.9, 0.923, 0.0, -0.0, 1.571))
	(atpose braiserlid#1 p10293=(0.567, 7.872, 0.712, 0.0, -0.0, 1.303))
	(atpose pepper-shaker p10296=(0.764, 7.303, 1.164, 0.0, -0.0, 3.142))
	(atpose chicken-leg p10292=(0.654, 4.846, 1.384, 0.0, 0.0, -0.366))

	(isopenedposition counter#1::chewie_door_left_joint pstn7440=-1.872)
	(isopenedposition counter#1::chewie_door_right_joint pstn7439=1.872)
	(isopenedposition fridge#1::fridge_door pstn7441=1.78)

	(aconf left aq896=(0.677, -0.343, 1.2, -1.467, 1.242, -1.954, 2.223))
	(aconf right aq288=(-0.677, -0.343, -1.2, -1.467, -1.242, -1.954, -2.223))

	(ataconf right aq288=(-0.677, -0.343, -1.2, -1.467, -1.242, -1.954, -2.223))
	(ataconf left aq896=(0.677, -0.343, 1.2, -1.467, 1.242, -1.954, 2.223))

	(contained salt-shaker p10295=(0.771, 7.071, 1.152, 0.0, -0.0, 3.142) counter#1::sektion)
	(contained pepper-shaker p10296=(0.764, 7.303, 1.164, 0.0, -0.0, 3.142) counter#1::sektion)

	(supported braiserbody#1 p10294=(0.7, 8.9, 0.923, 0.0, -0.0, 1.571) counter#1::indigo_tmp)
	(supported braiserlid#1 p10293=(0.567, 7.872, 0.712, 0.0, -0.0, 1.303) counter#1::front_left_stove)

  )

  (:goal (and
    (open fridge#1::fridge_door)
  ))
)
        