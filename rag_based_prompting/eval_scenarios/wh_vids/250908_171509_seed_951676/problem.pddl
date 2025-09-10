
(define
  (problem test_kitchen_chicken_soup_250908_171509_seed_951676)
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

	(canpull right)
	(canpull left)

	(cangrasphandle)

	(handempty right)
	(handempty left)

	(food chicken-leg)

	(controllable right)
	(controllable left)

	(region braiserbody#1)
	(region counter#1::indigo_tmp)
	(region counter#1::hitman_countertop)
	(region fridge#1::shelf_top)
	(region counter#1::sektion)
	(region counter#1::front_right_stove)
	(region braiserbody#1::braiser_bottom)
	(region counter#1::front_left_stove)
	(region basin#1::basin_bottom)

	(graspable chicken-leg)
	(graspable salt-shaker)
	(graspable braiserbody#1)
	(graspable braiserlid#1)
	(graspable pepper-shaker)

	(sprinkler pepper-shaker)
	(sprinkler salt-shaker)

	(space counter#1::sektion)
	(space braiserbody#1)

	(surface fridge#1::shelf_top)
	(surface counter#1::front_right_stove)
	(surface braiserbody#1::braiser_bottom)
	(surface braiserbody#1)
	(surface counter#1::front_left_stove)
	(surface counter#1::indigo_tmp)
	(surface basin#1::basin_bottom)
	(surface counter#1::hitman_countertop)

	(bconf q592=(2.0, 6.25, 0.2, 3.142))

	(atbconf q592=(2.0, 6.25, 0.2, 3.142))

	(staticlink counter#1::front_left_stove)
	(staticlink braiserbody#1)
	(staticlink counter#1::indigo_tmp)
	(staticlink basin#1::basin_bottom)
	(staticlink counter#1::hitman_countertop)
	(staticlink fridge#1::shelf_top)
	(staticlink counter#1::sektion)
	(staticlink counter#1::front_right_stove)
	(staticlink braiserbody#1::braiser_bottom)

	(door counter#1::chewie_door_right_joint)
	(door fridge#1::fridge_door)
	(door counter#1::chewie_door_left_joint)

	(joint counter#1::chewie_door_right_joint)
	(joint fridge#1::fridge_door)
	(joint oven#1::knob_joint_3)
	(joint faucet#1::joint_faucet_0)
	(joint oven#1::knob_joint_2)
	(joint counter#1::chewie_door_left_joint)
	(unattachedjoint faucet#1::joint_faucet_0)
	(unattachedjoint oven#1::knob_joint_2)
	(unattachedjoint counter#1::chewie_door_left_joint)
	(unattachedjoint counter#1::chewie_door_right_joint)
	(unattachedjoint fridge#1::fridge_door)
	(unattachedjoint oven#1::knob_joint_3)

	(containable braiserbody#1 counter#1::sektion)
	(containable chicken-leg counter#1::sektion)
	(containable salt-shaker counter#1::sektion)
	(containable braiserlid#1 counter#1::sektion)
	(containable chicken-leg braiserbody#1)
	(containable pepper-shaker counter#1::sektion)

	(atposition oven#1::knob_joint_3 pstn11962=0.0)
	(atposition faucet#1::joint_faucet_0 pstn11963=0.0)
	(atposition fridge#1::fridge_door pstn11960=1.78)
	(atposition counter#1::chewie_door_right_joint pstn11958=1.872)
	(atposition counter#1::chewie_door_left_joint pstn11959=-1.872)
	(atposition oven#1::knob_joint_2 pstn11961=0.0)

	(position faucet#1::joint_faucet_0 pstn11963=0.0)
	(position counter#1::chewie_door_right_joint pstn11958=1.872)
	(position counter#1::chewie_door_left_joint pstn11959=-1.872)
	(position oven#1::knob_joint_3 pstn11962=0.0)
	(position fridge#1::fridge_door pstn11960=1.78)
	(position oven#1::knob_joint_2 pstn11961=0.0)

	(isclosedposition oven#1::knob_joint_3 pstn11962=0.0)
	(isclosedposition oven#1::knob_joint_2 pstn11961=0.0)
	(isclosedposition faucet#1::joint_faucet_0 pstn11963=0.0)
	(stackable chicken-leg braiserbody#1::braiser_bottom)
	(stackable salt-shaker counter#1::indigo_tmp)
	(stackable pepper-shaker counter#1::indigo_tmp)
	(stackable chicken-leg basin#1::basin_bottom)
	(stackable braiserbody#1 counter#1::indigo_tmp)
	(stackable chicken-leg counter#1::indigo_tmp)
	(stackable braiserlid#1 braiserbody#1)
	(stackable braiserlid#1 counter#1::front_left_stove)
	(stackable braiserlid#1 basin#1::basin_bottom)
	(stackable braiserlid#1 counter#1::hitman_countertop)
	(stackable salt-shaker counter#1::hitman_countertop)
	(stackable braiserlid#1 counter#1::indigo_tmp)
	(stackable pepper-shaker counter#1::hitman_countertop)

	(isjointto counter#1::chewie_door_right_joint counter#1)
	(isjointto counter#1::chewie_door_left_joint counter#1)
	(isjointto oven#1::knob_joint_2 oven#1)
	(isjointto fridge#1::fridge_door fridge#1)
	(isjointto faucet#1::joint_faucet_0 faucet#1)
	(isjointto oven#1::knob_joint_3 oven#1)

	(pose salt-shaker p23394=(0.771, 7.071, 1.152, 0.0, -0.0, 3.142))
	(pose braiserbody#1 p23390=(0.7, 8.783, 0.923, 0.0, -0.0, 1.571))
	(pose chicken-leg p23391=(0.654, 4.846, 1.384, 0.0, 0.0, -0.366))
	(pose braiserlid#1 p23392=(0.567, 7.872, 0.712, 0.0, -0.0, 1.625))
	(pose pepper-shaker p23395=(0.764, 7.303, 1.164, 0.0, -0.0, 3.142))

	(atpose braiserlid#1 p23392=(0.567, 7.872, 0.712, 0.0, -0.0, 1.625))
	(atpose pepper-shaker p23395=(0.764, 7.303, 1.164, 0.0, -0.0, 3.142))
	(atpose salt-shaker p23394=(0.771, 7.071, 1.152, 0.0, -0.0, 3.142))
	(atpose braiserbody#1 p23390=(0.7, 8.783, 0.923, 0.0, -0.0, 1.571))
	(atpose chicken-leg p23391=(0.654, 4.846, 1.384, 0.0, 0.0, -0.366))

	(aconf left aq816=(0.677, -0.343, 1.2, -1.467, 1.242, -1.954, 2.223))
	(aconf right aq608=(-0.677, -0.343, -1.2, -1.467, -1.242, -1.954, -2.223))
	(isopenedposition counter#1::chewie_door_left_joint pstn11959=-1.872)
	(isopenedposition counter#1::chewie_door_right_joint pstn11958=1.872)
	(isopenedposition fridge#1::fridge_door pstn11960=1.78)

	(ataconf left aq816=(0.677, -0.343, 1.2, -1.467, 1.242, -1.954, 2.223))
	(ataconf right aq608=(-0.677, -0.343, -1.2, -1.467, -1.242, -1.954, -2.223))

	(contained pepper-shaker p23395=(0.764, 7.303, 1.164, 0.0, -0.0, 3.142) counter#1::sektion)
	(contained salt-shaker p23394=(0.771, 7.071, 1.152, 0.0, -0.0, 3.142) counter#1::sektion)

	(supported braiserbody#1 p23393=(0.7, 8.783, 0.923, 0.0, -0.0, 1.571) counter#1::indigo_tmp)
	(supported braiserlid#1 p23392=(0.567, 7.872, 0.712, 0.0, -0.0, 1.625) counter#1::front_left_stove)

  )

  (:goal (and
    (open fridge#1::fridge_door)
  ))
)
        