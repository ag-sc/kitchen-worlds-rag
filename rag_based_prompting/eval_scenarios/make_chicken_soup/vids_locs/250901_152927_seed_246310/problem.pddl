
(define
  (problem test_kitchen_chicken_soup_250901_152927_seed_246310)
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

	(graspable braiserlid#1)
	(graspable pepper-shaker)
	(graspable chicken-leg)
	(graspable salt-shaker)
	(graspable braiserbody#1)

	(sprinkler pepper-shaker)
	(sprinkler salt-shaker)

	(space counter#1::sektion)
	(space braiserbody#1)
	(staticlink braiserbody#1)
	(staticlink braiserbody#1::braiser_bottom)
	(staticlink counter#1::front_left_stove)
	(staticlink basin#1::basin_bottom)
	(staticlink counter#1::indigo_tmp)
	(staticlink counter#1::front_right_stove)
	(staticlink counter#1::hitman_countertop)
	(staticlink fridge#1::shelf_top)
	(staticlink counter#1::sektion)

	(door fridge#1::fridge_door)
	(door counter#1::chewie_door_left_joint)
	(door counter#1::chewie_door_right_joint)

	(bconf q880=(2.0, 6.25, 0.2, 3.142))
	(region counter#1::front_left_stove)
	(region basin#1::basin_bottom)
	(region counter#1::indigo_tmp)
	(region counter#1::front_right_stove)
	(region counter#1::hitman_countertop)
	(region fridge#1::shelf_top)
	(region counter#1::sektion)
	(region braiserbody#1)
	(region braiserbody#1::braiser_bottom)

	(atbconf q880=(2.0, 6.25, 0.2, 3.142))
	(surface counter#1::front_right_stove)
	(surface counter#1::hitman_countertop)
	(surface braiserbody#1)
	(surface fridge#1::shelf_top)
	(surface counter#1::front_left_stove)
	(surface braiserbody#1::braiser_bottom)
	(surface counter#1::indigo_tmp)
	(surface basin#1::basin_bottom)
	(unattachedjoint oven#1::knob_joint_2)
	(unattachedjoint counter#1::chewie_door_left_joint)
	(unattachedjoint counter#1::chewie_door_right_joint)
	(unattachedjoint faucet#1::joint_faucet_0)
	(unattachedjoint fridge#1::fridge_door)
	(unattachedjoint oven#1::knob_joint_3)

	(isjointto fridge#1::fridge_door fridge#1)
	(isjointto counter#1::chewie_door_right_joint counter#1)
	(isjointto oven#1::knob_joint_3 oven#1)
	(isjointto oven#1::knob_joint_2 oven#1)
	(isjointto counter#1::chewie_door_left_joint counter#1)
	(isjointto faucet#1::joint_faucet_0 faucet#1)
	(joint counter#1::chewie_door_right_joint)
	(joint faucet#1::joint_faucet_0)
	(joint fridge#1::fridge_door)
	(joint oven#1::knob_joint_2)
	(joint oven#1::knob_joint_3)
	(joint counter#1::chewie_door_left_joint)

	(containable pepper-shaker counter#1::sektion)
	(containable braiserbody#1 counter#1::sektion)
	(containable chicken-leg counter#1::sektion)
	(containable salt-shaker counter#1::sektion)
	(containable chicken-leg braiserbody#1)
	(containable braiserlid#1 counter#1::sektion)

	(atposition oven#1::knob_joint_2 pstn10178=0.0)
	(atposition counter#1::chewie_door_left_joint pstn10176=-1.872)
	(atposition counter#1::chewie_door_right_joint pstn10175=1.872)
	(atposition faucet#1::joint_faucet_0 pstn10180=0.0)
	(atposition oven#1::knob_joint_3 pstn10179=0.0)
	(atposition fridge#1::fridge_door pstn10177=1.78)
	(position fridge#1::fridge_door pstn10177=1.78)
	(position oven#1::knob_joint_2 pstn10178=0.0)
	(position counter#1::chewie_door_left_joint pstn10176=-1.872)
	(position counter#1::chewie_door_right_joint pstn10175=1.872)
	(position faucet#1::joint_faucet_0 pstn10180=0.0)
	(position oven#1::knob_joint_3 pstn10179=0.0)

	(stackable braiserlid#1 counter#1::hitman_countertop)
	(stackable pepper-shaker counter#1::indigo_tmp)
	(stackable braiserlid#1 braiserbody#1)
	(stackable chicken-leg braiserbody#1::braiser_bottom)
	(stackable pepper-shaker counter#1::hitman_countertop)
	(stackable braiserbody#1 counter#1::indigo_tmp)
	(stackable chicken-leg counter#1::indigo_tmp)
	(stackable salt-shaker counter#1::indigo_tmp)
	(stackable chicken-leg basin#1::basin_bottom)
	(stackable braiserlid#1 counter#1::front_left_stove)
	(stackable salt-shaker counter#1::hitman_countertop)
	(stackable braiserlid#1 counter#1::indigo_tmp)
	(stackable braiserlid#1 basin#1::basin_bottom)

	(isclosedposition faucet#1::joint_faucet_0 pstn10180=0.0)
	(isclosedposition oven#1::knob_joint_3 pstn10179=0.0)
	(isclosedposition oven#1::knob_joint_2 pstn10178=0.0)

	(pose braiserbody#1 p31554=(0.7, 9.0, 0.923, 0.0, -0.0, 1.571))
	(pose chicken-leg p31555=(0.654, 4.846, 1.384, 0.0, 0.0, -0.366))
	(pose pepper-shaker p31559=(0.764, 7.303, 1.164, 0.0, -0.0, 3.142))
	(pose braiserlid#1 p31556=(0.567, 7.872, 0.712, 0.0, -0.0, 2.612))
	(pose salt-shaker p31558=(0.771, 7.071, 1.152, 0.0, -0.0, 3.142))

	(atpose braiserbody#1 p31554=(0.7, 9.0, 0.923, 0.0, -0.0, 1.571))
	(atpose chicken-leg p31555=(0.654, 4.846, 1.384, 0.0, 0.0, -0.366))
	(atpose braiserlid#1 p31556=(0.567, 7.872, 0.712, 0.0, -0.0, 2.612))
	(atpose pepper-shaker p31559=(0.764, 7.303, 1.164, 0.0, -0.0, 3.142))
	(atpose salt-shaker p31558=(0.771, 7.071, 1.152, 0.0, -0.0, 3.142))

	(aconf left aq392=(0.677, -0.343, 1.2, -1.467, 1.242, -1.954, 2.223))
	(aconf right aq936=(-0.677, -0.343, -1.2, -1.467, -1.242, -1.954, -2.223))
	(isopenedposition counter#1::chewie_door_right_joint pstn10175=1.872)
	(isopenedposition fridge#1::fridge_door pstn10177=1.78)
	(isopenedposition counter#1::chewie_door_left_joint pstn10176=-1.872)

	(ataconf left aq392=(0.677, -0.343, 1.2, -1.467, 1.242, -1.954, 2.223))
	(ataconf right aq936=(-0.677, -0.343, -1.2, -1.467, -1.242, -1.954, -2.223))

	(contained salt-shaker p31558=(0.771, 7.071, 1.152, 0.0, -0.0, 3.142) counter#1::sektion)
	(contained pepper-shaker p31559=(0.764, 7.303, 1.164, 0.0, -0.0, 3.142) counter#1::sektion)

	(supported braiserlid#1 p31556=(0.567, 7.872, 0.712, 0.0, -0.0, 2.612) counter#1::front_left_stove)
	(supported braiserbody#1 p31557=(0.7, 9.0, 0.923, 0.0, -0.0, 1.571) counter#1::indigo_tmp)

  )

  (:goal (and
    (open fridge#1::fridge_door)
  ))
)
        