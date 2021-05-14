Changelog
=========


(unreleased)
------------
- Updated tutorial with an example single cell sim (without network)
  [Anil Tuncel]
- Removed unreachable bluepy<=0.16.0 branch. [Anil Tuncel]

  Patch 2: setup.py bluepy remove bbp and  add brion

  it's unreachable since setup.py assumes bluepy>2.1.0
- Added numpy and matplotlib dependencies. [Anil Tuncel]
- Replace methodtools with cachetools to reduce dependencies. [Anil
  Tuncel]

  as methodtools depend on wirerope and inspect2 #BGLPY-80
  Patch 2: remove the cache of is_cell_target method
  since it's cheap
- Remove extra[bbp] since brion is in install_requires. [Anil Tuncel]

  *PATCH1: bluepy>=2.1.0.dev6 -> bluepy>=2.1.0
- Drop deprecated bluepy.v2 subpackage. [Anil Tuncel]
- Merge "Merge branch 'warnings'" [Anil Tuncel]
- Merge branch 'warnings' [Anil Tuncel]
- Merge "error message made more informative" [Anil Tuncel]
- Error message made more informative. [Anil Tuncel]

  ignore_populationid_error=True is mentioned
- Changelog update upon updating the tag. [Anil Tuncel]
- Use methodtools lru cache to prevent memory leak. [Werner Van Geit]
- Apply the sonata spike report update [BLPY-244] [Anil Tuncel]

  * apply renaming of brian->brion
- Added FAQ page with MPT ERROR: PMI2_Init. [Anil Tuncel]
- Documentation update. [Anil Tuncel]

  * mention missing parameters in docstring
  * refer to the jupyter notebook in insilico-cookbook
    in the tutorial section.
  * add changelog to sphinx.
  * PATCH 2: added docs/source/changelog.rst
- Added option to use hoc with AIS_scaler, aligned with
  https://bbpcode.epfl.ch/code/#/c/52044/ [arnaudon]

  * rebase master
  * added changelog entry
  * rebase master at ffc293a bluepy v1.0.0 integration
- Bluepy v1.0.0 integration. [Anil Tuncel]

  * PATCH 2: BLPSynapse.POST_SEGMENT_ID for newer bluepy
  * PATCH 3: Added changelog entry
- Updated docstring for Synapse.synid to contain tuple idx info. [Anil
  Tuncel]
- Apply pep8 code style with E501,W504,W503,E741 ignored. [Anil Tuncel]

  Tox & Jenkins plans are updated accordingly
- Read synapse locations from SONATA field and round synapse delays to
  timestep. [Sirio Bolaños Puchet]

  * style: line lengths decreased to 80
- Remove bluepy 'sonata' extra in version >=0.16.0 [BGLPY-78] [Anil
  Tuncel]

  * also remove the explicit h5py<3.0.0 dependency since bluepy handles it
- Merge "Add support for MinisSingleVesicle, SpikeThreshold, V_Init,
  Celsius" [Anil Tuncel]
- Add support for MinisSingleVesicle, SpikeThreshold, V_Init, Celsius.
  [Sirio Bolaños Puchet]

  * Added a gpfs test
  * added a custom exception
  * rebased master branch
  * used get_mainsim_voltage with t_start, t_stop, t_step parameters in the test
  * CHANGELOG updated
- Tests remove unnecessary ssim object creations. [Anil Tuncel]
- Use absolute paths in blueconfigs [BLPY-178] [Anil Tuncel]

  * adapted the tests accordingly
- Remove python27 from jenkins plan. [Anil Tuncel]
- Introducing t_start, t_stop, t_step parameters for
  get_mainsim_voltage_trace. [Anil Tuncel]

  The motivation is due to the performance.
  Retrieving the mainsim voltage using bluepy on large simulations takes very long.
  With the use of bluepy api v2 this change enables retrieving only a section of voltage rather than the entire simulation voltage.
- Use h5py<3.0.0. [Anil Tuncel]

  h5py 3.0.0 is parsing the dtype (previously parsed as str) as bytes.
  There may be other changed datatypes as well.
  Until a long-term solution can be found, it's best to pin the version down.
- Pin version of pyrsistent in tox. [Werner Van Geit]
- Moved download = true in tox.ini. [Werner Van Geit]
- Trying to avoid pinning virtualenv. [Werner Van Geit]
- Removed pyrsistent dependency since it became a dependency of bluepy.
  [Anil Tuncel]

  * in tox use download=true to get the recent pip that comes with a new dependency resolver
  * remove unused pandas dependency
  * removed the old bluepy-configfile-0.1.2.dev1 version dependency (bluepy already has bluepy-configfile>=0.1.11)
- Setting RNGSettings.mode to automatically set neuron.h.rngMode. [Anil
  Tuncel]

  This implementation is based on bglpy-68 issue.
  The purpose is to behave the same as neurodamus does.
  * Made RNGSettings a singleton class since it's dealing with a global variable.
- Make sure targets used by _evaluate_connection_parameters exist. [Anil
  Tuncel]
- Noisestim_count to be incremented whether or not it's applied to the
  gid. [Anil Tuncel]

  see #bglpy69 for further info
- Merge branch 'master' of ssh://bbpcode.epfl.ch/sim/BGLibPy. [Werner
  Van Geit]
- Fix synapse_detail error when add_minis is False and synapse_detail>0.
  [Anil Tuncel]
- Fix idiotic warning thrown by python lately. [Werner Van Geit]


4.4 (2020-09-21)
----------------
- Change behavior delayed connection blocks, weight is now a scaler
  instead of absolute value. [Werner Van Geit]
- Enforce pyrsistent<0.17.1 for the py27 build. [Anil Tuncel]

  pyrsistent requires python>3.5 from that version on
- Fix printv in synapses.py. [Werner Van Geit]
- When get time trace from mainsim, only look at 1 gid to save memory.
  [Werner Van Geit]
- Removed all_targets_dict, since it uses too much memory in new
  circuits, replaced with direct bluepy call and lru_cache. [Werner Van
  Geit]
- Added explicit delete() method to ssim. [Werner Van Geit]
- Added a setting to ignore missing population id in projection blocks.
  [Anil Tuncel]

  * added docstring for rng_mode in SSim constructor
  * added a module for custom exceptions
  * rename: ignore_missing_populationid -> ignore_populationid_error
- Added support for MorphologyType field in BlueConfig. [Werner Van
  Geit]
- Small fix of typo that shouldn't affect output. [Werner Van Geit]
- Use analytical solution for hill coefficient. [Werner Van Geit]
- Add support for a* targets in connections. [Werner Van Geit]
- Merge "vectorised usage of Bluepy api for get_sonata_mecombo_emodels"
  [Werner Van Geit]
- Vectorised usage of Bluepy api for get_sonata_mecombo_emodels. [Anil
  Tuncel]

  * bc_circuit.cells.get use None to get all cells
  * don't use mecombo_emodels dict if node_properties_available
  * get_sonata_mecombo_emodels to return 2 dicts for threshold and holding currs
- Merge "use issubset for checking node properties" [Werner Van Geit]
- Use issubset for checking node properties. [Anil Tuncel]
- Made thalamus test trace shorter, removed 1st time point until we
  understand change in ND. [Werner Van Geit]
- Merge "added sonata nodes.h5 support" [Werner Van Geit]
- Added sonata nodes.h5 support. [Anil Tuncel]

  * updated changelog
  * get_sonata_mecombo_emodels to extract nodes.h5 properties
  * node_properties_available to check if nodes.h5 can be used
  * setup.py to use bluepy[sonata]>=0.14.12
  * merged ssim changes on sonata branch
  * get_sonata_mecombo_emodels indentation fix after merge
- Added thalamus tests to jenkins plan * change the thalamus test path
  to the recently run ND simulation below. /gpfs/bbp.cscs.ch/project/pro
  j55/tuncel/simulations/release/2020-08-06-v2/bglibpy-thal-test-with-
  projections. [Anil Tuncel]
- Use nosepipe to isolate tests. [Werner Van Geit]
- Fix lru_cache in python2. [Werner Van Geit]
- Isolating nose tests. [Werner Van Geit]


4.3 (2020-08-05)
----------------
- Fixing sonata properties check. [Werner Van Geit]
- Use bluepy available_properties, no need to check h5 version anymore.
  [Werner Van Geit]
- Added reading of inh/exc minis freq from nodes file, use hill
  coefficients and cond ratios from nodes file. [Werner Van Geit]
- Merge changes from topic 'remove-unused' [Werner Van Geit]

  * changes:
    removed unused tests depending on the data that no longer exist
    removed unused psp_bglib test directory
- Removed unused tests depending on the data that no longer exist. [Anil
  Tuncel]

  These tests used to depend on the data stored at /bgscratch
- Removed unused psp_bglib test directory. [Anil Tuncel]

  The code here cannot be executed since the directories to the config files no longer exist
- Corrected rst link. [Anil Tuncel]
- Update dependencies: mention rpm and deb packages for python compiled
  neurons. [Anil Tuncel]
- BGLibPy tutorial is updated. [Anil Tuncel]

  Changelog:
  * Tutorial to use an existing BlueConfig file from the examples directory
  * Mention of paired simulations via PSP validation
  * Code block is added to enable spontMinis and synapses
- Temporary fix for documentation theme failing. [Andrew Hale]
- Removed Python 2.7 usage suggestion. [Anil Tuncel]
- Updated dependencies docs. [Anil Tuncel]
- Removed viz cluster info. [Anil Tuncel]
- Merged .gitignores. [Anil Tuncel]
- Merge changes from topic 'small-fixes' [Werner Van Geit]

  * changes:
    using not to check if dict is empty
    string comparison to literal use ==
    compare the string value, not its reference
- Using not to check if dict is empty. [Anil Tuncel]

  Before it was compared to an empty list
- String comparison to literal use == [Anil Tuncel]
- Compare the string value, not its reference. [Anil Tuncel]
- Removed empty lines. [Anil Tuncel]
- Removed spontminis_set flag. [Anil Tuncel]
- Removed the default value for SpontMinis. [Anil Tuncel]
- In case of multiple spontminis take the latest. [Tuncel Anil]
- Updated .gitignore. [Tuncel Anil]
- Merge changes from topic 'test_thalamus' [Werner Van Geit]

  * changes:
    added test for thalamus The simulation contains multiple projections and stimuli
    restrict the compilation of neocortexv5 to test&v5 It is not needed to be compiled for the other settings. When thalamus tests are introduced it should not be compiled for those
- Added test for thalamus The simulation contains multiple projections
  and stimuli. [Tuncel Anil]
- Restrict the compilation of neocortexv5 to test&v5 It is not needed to
  be compiled for the other settings. When thalamus tests are introduced
  it should not be compiled for those. [Tuncel Anil]
- Downgrading virtualenv on ubuntu 16.04. [Werner Van Geit]
- Try older nrn commit. [Werner Van Geit]
- Fix git checkout. [Werner Van Geit]
- Trying build with other nrn commit. [Werner Van Geit]
- Cloning neuron deeper. [Werner Van Geit]
- Pull older version of neuron for testing. [Werner Van Geit]
- Removed unnecessary cp operations from install_neurodamus. [Tuncel
  Anil]
- BUGFIX: check&remove NRRP using the Enum value Other were getting
  removed before in case of multiple projections, since the check was
  missing. [Tuncel Anil]
- Fix class and module docs. [Andrew Hale]

  Class and module documentation was being generated, however it
  was not linked anywhere that was useful on the docs pages.
  This commit cleans up some code that was required with older
  versions of sphinx.

  This commit puts all class/module documentation on the same
  page as the class/module itself.
- Fixing v5 tests. [Werner Van Geit]
- Fixing tests. [Werner Van Geit]


4.2 (2019-10-24)
----------------
- Fix target_popid in synapse. [Werner Van Geit]
- Changes related to minis with projections. [Werner Van Geit]
- Switch to BBP doc theme. [Werner Van Geit]
- Handle case with no patch version in bglibpy version. [Werner Van
  Geit]
- Libsonata is now a dependency. [Werner Van Geit]
- Remove versions.py which is a relic from the past. [Werner Van Geit]
- Surround synapseconf statements by {} [Werner Van Geit]
- Remove unused libs in upload_docs. [Werner Van Geit]


4.1 (2019-08-06)
----------------
- Change the synids provided by bluepy so that they match nd. [Werner
  Van Geit]
- Merge branch 'master' of ssh://bbpcode.epfl.ch/sim/BGLibPy. [Werner
  Van Geit]
- Use new options for uploading docs. [Andrew Hale]

  Utilise options from docs-internal-upload to manage
  uploading docs (or not) depending on whether they are duplicates.

  Requires docs-internal-upload>=0.0.8
- Pass USER env variable to tox envs. [Andrew Hale]
- Use docs-internal-upload for docs release. [Andrew Hale]

  Transition the upload of documentation to use the
  docs-internal-upload package. This simplifies the logic
  in .upload_docs.py and removes any need for interacting
  with the docs repo directly.
- Add depth to neurodamus core clone. [Werner Van Geit]
- Remove vangeit from neurodamus download. [Werner Van Geit]
- Finalized move to nd core. [Werner Van Geit]
- Switching to neurodamus core. [Werner Van Geit]
- Improved importer, bglibpy_modlib_path can now be list. [Werner Van
  Geit]
- Remove presynaptic location request to bluepy. [Werner Van Geit]
- Merge branch 'master' into add_projections. [Werner Van Geit]
- Extend numpy encoder for json in python3. [Werner Van Geit]
- Lowered precision of some tests because of change in nrnsim repo.
  [Werner Van Geit]
- Make sure we have absolute path of doc html dir. [Werner Van Geit]
- Fix for hocobjects not having len() in new nrn release. [Werner Van
  Geit]
- Add a projections field to ssim instantantie gid. [Werner Van Geit]
- Temporarily pin version of tox to make tests work. [Werner Van Geit]
- Small text edit. [Werner Van Geit]
- Update package version. [Werner Van Geit]
- Merge branch 'master' of ssh://bbpcode.epfl.ch/sim/BGLibPy. [Werner
  Van Geit]
- Fix verbose level from env. [Werner Van Geit]
- Fix syn id iterator in ssim. [Werner Van Geit]
- Add numpy encoder to convert dict to json string. [Werner Van Geit]
- Add default rng mode. [Werner Van Geit]
- Fix issue in previous commit (nrrp check) [Werner Van Geit]
- Add test for non-integer nrrp values. [Werner Van Geit]
- Fix sonata test in ssim. [Werner Van Geit]
- Raise exception when section with particual isec not found. [Werner
  Van Geit]
- Add check for sonata connectome, switch nrrp behavior based on it.
  [Werner Van Geit]
- Add hack to handle situation where ascii subdir doesnt' exist. [Werner
  Van Geit]
- Catch indexerror when no threshold/holding current value found.
  [Werner Van Geit]
- Fix python title in doc. [Werner Van Geit]
- Add python 3 version to classifiers in setup.py. [Werner Van Geit]
- Make v5 test py3 compatible. [Werner Van Geit]
- Remove 'vangeit' from neurodamus clone. [Werner Van Geit]
- Make BGLibPy python3 compatible. [Werner Van Geit]


4.0 (2018-11-26)
----------------
- Bumping version. [Werner Van Geit]


3.3 (2018-11-26)
----------------
- Merge branch 'master' of ssh://bbpcode.epfl.ch/sim/BGLibPy. [Werner
  Van Geit]
- Fixed access to proj_nrn.h5 files. [Arseny V. Povolotsky]
- Fixing init of neurodamus in importer after changes in neurodamus
  master. [Werner Van Geit]
- Enable verbose tox in jenkins. [Werner Van Geit]
- Remove mpi file from neurodamus. [Werner Van Geit]
- Finalize tests vclamp, add doc. [Werner Van Geit]
- Add new add_voltage_clamp method. [Werner Van Geit]
- Added BGLIBPY_VERBOSE_LEVEL env variable. [Werner Van Geit]
- Fix python3 change in Neuron. [Werner Van Geit]
- Add ttx flag to tools.holding_current() [Werner Van Geit]
- Fix last commit in case CircuitConfig is used instead of BlueConfig.
  [Werner Van Geit]
- Set neuron tstop in constructor of ssim because it used in TStim.hoc.
  [Werner Van Geit]
- Merge branch 'master' of ssh://bbpcode.epfl.ch/sim/BGLibPy. [Werner
  Van Geit]
- Open nrn.h5 in read-only mode. [Arseny V. Povolotsky]
- Force downgrade sphinx to avoid bug in latest sphinx release. [Werner
  Van Geit]
- Temporariy disable 1 test because circuit disappeared. [Werner Van
  Geit]
- Small fix in .jenkins.sh. [Werner Van Geit]
- Upload docs only on BB5. [Werner Van Geit]
- Run gpfs tests on BB5 in jenkins. [Werner Van Geit]
- Remove pybinreports from setup.py requirements. [Werner Van Geit]
- Read the nrn.h5 version from bglibpy instead of counting on bluepy.
  [Werner Van Geit]
- Introduce get_time_trace and get_voltage_trace that return pos times.
  [Werner Van Geit]
- Fixing case where hypamp is empty in tsv file, for hippocampus.
  [Werner Van Geit]
- Random123 fixes. [Werner Van Geit]
- Merge branch 'master' into add_random123. [Werner Van Geit]
- Ignore error when we can't upload do release devpi. [Werner Van Geit]
- Add verbose message to add_replay_hypamp. [Werner Van Geit]
- Unpin Brain version, a bug has been fixed. [Werner Van Geit]
- Also upload package to devpi release. [Werner Van Geit]
- Add pybinreports to bbp extra. [Werner Van Geit]
- Upload docs and devpi from cscs viz instead of ubuntu. [Werner Van
  Geit]
- Fall back to version 2.1.0 of Brain because of a bug in Brain. [Werner
  Van Geit]
- Import RNGSettings.hoc, also remove version number from brain
  dependency. [Werner Van Geit]
- Add bbp extra to tox.ini. [Werner Van Geit]
- Moved brain dependency to [bbp] extra. [Werner Van Geit]
- More small doc fixes. [Werner Van Geit]
- More doc fixes. [Werner Van Geit]
- Fixes in documentation. [Werner Van Geit]
- Add seeds to synapses, minis, etc. [Werner Van Geit]
- Adding rngsettings argument to synapse. [Werner Van Geit]
- Added new rngsettings class. [Werner Van Geit]
- Pin version of Brain to avoid bug in devpi package. [Werner Van Geit]
- Fix warning about pandas indexing. [Werner Van Geit]
- Fixing synapse ids when intersect_pre_gids is used. [Werner Van Geit]
- Make sure add_synapses is set to true if pre_spike_trains are
  specified. [Werner Van Geit]
- Add a pre_spike_trains and projection option to instantiate_gids.
  [Werner Van Geit]
- Update doc to solve nix trouble. [Werner Van Geit]
- Implement change in neurodamus that puts synapses at 0.99.. and
  0.00..1. [Werner Van Geit]
- Add 1 more spot check to make sure nrrp value I get is correct.
  [Werner Van Geit]
- Implementing getting threshold/holding from tsv and adding v6 test.
  [Werner Van Geit]
- Add default implementation of enable/disable ttx. [Werner Van Geit]
- First version that runs (unvalidated) with Nrrp read from nrn.h5.
  [Werner Van Geit]
- Fix for MVR nrrp. [Werner Van Geit]
- Add functionality to tools.holding_current to manage v6 templates.
  [Werner Van Geit]
- Change how templates are loaded, in ssim, assume hoc has correct
  morph. [Werner Van Geit]
- Fix tests that use circuits on gpfs. [Werner Van Geit]


3.2 (2017-11-08)
----------------
- First version of code that reads nrrp var from nrn.h5 (unvalidated)
  [Werner Van Geit]
- Mention new way of using NEURON nix on CSCS viz in doc. [Werner Van
  Geit]
- Remove modlibpath warning, it confuses people. [Werner Van Geit]
- Access 'OutputRoot' config key only when needed. [Arseny V.
  Povolotsky]
- Mention --enable-unicode=ucs4 python compilation problem in doc.
  [Werner Van Geit]
- Fix small things in doc. [Werner Van Geit]
- Merge branch 'remove_cmake' [Werner Van Geit]
- Fixed link to dep section in docs. [Werner Van Geit]
- Improve installation docs. [Werner Van Geit]
- Small renaming in test_ssim. [Werner Van Geit]
- Update README about how to recreate neurodamus test sims. [Werner Van
  Geit]
- Remove soma2h5 script. [Werner Van Geit]
- Add mvr test, also rerun all neurodamus test sims. [Werner Van Geit]
- Refactor code to generate test sims using neurodamus. [Werner Van
  Geit]
- Reran all neurodamus simulations, removed all soma.h5 files. [Werner
  Van Geit]
- Remove all CMakeLists.txt. [Werner Van Geit]
- Changed doc upload string, add py3 tox target. [Werner Van Geit]
- Added test for threshold current in proj64, still disabled for now.
  [Werner Van Geit]
- Let tox pass https_proxy variable. [Werner Van Geit]
- Add git proxy to .jenkins.sh. [Werner Van Geit]
- Recreate tox env in jenkins. [Werner Van Geit]
- Use github neuron instead of 'official' release for testing. [Werner
  Van Geit]
- Fix importer warning message. [Werner Van Geit]
- Reenable some complicated gpfs tests. [Werner Van Geit]
- Remove the 'recreate' from tox. [Werner Van Geit]
- Raise exception in connection when pre_spiketrain has negative time.
  [Werner Van Geit]
- Add mode for older cell templates. [Werner Van Geit]
- Enable proj64 test. [Werner Van Geit]
- Remove png-files delete from Makefile. [Werner Van Geit]
- Include hour:minutes in build time of sphinx doc. [Werner Van Geit]
- Fixing back-and-forth bluepy api changes. [Werner Van Geit]
- Fix destructor of ssim in case 'cells' doesn't exist. [Werner Van
  Geit]
- Changed permission of .jenkins.sh. [Werner Van Geit]
- Add jenkins shell script. [Werner Van Geit]
- Incorporate fixes for bugs in bluepy.v2. [Werner Van Geit]
- Remove code that removes all old docs. [Werner Van Geit]
- Remove old docs. [Werner Van Geit]
- Fix version on doc server. [Werner Van Geit]
- Small fixes in doc_upload. [Werner Van Geit]
- Store all major.minor versions on doc server. [Werner Van Geit]
- Prevent uploading same doc dir twice. [Werner Van Geit]
- Fix doc metadata fields. [Werner Van Geit]
- Fix order in tox.ini again. [Werner Van Geit]
- Using config to register email again. [Werner Van Geit]
- Add bbprelman email address to commit. [Werner Van Geit]
- Print git log during doc upload. [Werner Van Geit]
- Print git log in upload_doc. [Werner Van Geit]
- Cleanup upload_docs. [Werner Van Geit]
- Clean old doc from jekyll before uploading new. [Werner Van Geit]
- Fix devpi in tox. [Werner Van Geit]
- Switch to zip for devpi. [Werner Van Geit]
- More fixes in jekyll template. [Werner Van Geit]
- Fix jekyll template. [Werner Van Geit]
- Python to other file for upload2repo. [Werner Van Geit]
- Whitelisting upload2repo. [Werner Van Geit]
- Add bbprelman email address to upload doc script. [Werner Van Geit]
- Call python to run upload doc script. [Werner Van Geit]
- Remove -Q from sphinx build. [Werner Van Geit]
- Made doc upload more verbose. [Werner Van Geit]
- Change order of test/doc in tox. [Werner Van Geit]
- Add push master to doc upload. [Werner Van Geit]
- Added doc upload target. [Werner Van Geit]
- Upload to dev devpi instead of release. [Werner Van Geit]
- Add test-gpfs target. [Werner Van Geit]
- Update setup.py metadata. [Werner Van Geit]
- Make HOC_LIBRARY_PATH not found an exception. [Werner Van Geit]
- Remove dist dir before building sdist. [Werner Van Geit]
- Test for HOC_LIBRARY_PATH in importer. [Werner Van Geit]
- Add devpi target, started doc target. [Werner Van Geit]
- Add manifest file. [Werner Van Geit]
- Added versioneer versions. [Werner Van Geit]
- Fix yet another typo in package.json. [Liesbeth Vanherpe]
- Fix another typo in package.json. [Liesbeth Vanherpe]
- Fix typo in package.json. [Liesbeth Vanherpe]
- Fix package.json: switched fields. [Liesbeth Vanherpe]


3.1 (2017-10-06)
----------------
- Disable wget output when installing neuron, writing to log file.
  [Werner Van Geit]
- Use bluepy spikereport to parse out.dat. [Werner Van Geit]
- Reenable wget output in install neuron. [Werner Van Geit]
- Call tox with -v in Makefile. [Werner Van Geit]
- Fix test target makefile. [Werner Van Geit]
- Merge branch 'remove_cmake' of ssh://bbpcode.epfl.ch/sim/BGLibPy into
  remove_cmake. [Werner Van Geit]
- Bump version. [Werner Van Geit]
- First working version with new bluepy. [Werner Van Geit]
- Merge branch 'master' into remove_cmake. [Werner Van Geit]
- Updated package.json: needs patch version filled in. [Liesbeth
  Vanherpe]
- Updated package.json. [Liesbeth Vanherpe]
- Added metadata (package.json) for documentation purposes. [Liesbeth
  Vanherpe]
- Fix setup.py.in. [Liesbeth Vanherpe]
- Switch Documentation dir to jekylltest. [Werner Van Geit]
- Fixing doc_upload. [Werner Van Geit]
- Updated metadata for documentation purposes. [Liesbeth Vanherpe]
- Make long name in test a bit longer. [Werner Van Geit]
- Add test template for long name test. [Werner Van Geit]
- Short template name if too long. [Werner Van Geit]
- Ramove cmake installer, switch to pip. [Werner Van Geit]
- Showing bluepy version in exception added in last commit. [Werner Van
  Geit]
- Merge branch 'master' of ssh://bbpcode.epfl.ch/sim/BGLibPy. [Werner
  Van Geit]
- Add exception for ttx to make_passive. [Werner Van Geit]
- Add check for version BluePy and message why not to use >=0.10.0.
  [Werner Van Geit]
- Removed some useless print statements. [Werner Van Geit]
- Fixing holding_current() in test_tools to accommodate non-backward-
  compatible changes in BluePy. [Werner Van Geit]
- Added use_random123_stochkv option to simulator. [Werner Van Geit]
- Fixed create example doc. [Werner Van Geit]
- Reran regression tests after fix in Neurodamus regarding tsyn global
  var. [Werner Van Geit]
- Make sure /bgscratch isn't referenced. [Mike Gevaert]

  * some of the jenkins tests nodes have issues w/
    nfs, so don't let the tests even lookup /bgscratch
  * add .gitreview file
- Added BG/Q target in CMake. [Werner Van Geit]
- Fixed issue when user specified synapse_detail=2 and add_minis=False.
  [Werner Van Geit]
- One more pylint fix. [Werner Van Geit]
- Pylint fixes. [Werner Van Geit]
- Updating regression tests to work with fix in Neurodamus train() /
  Pulse function
  https://bbpteam.epfl.ch/project/issues/browse/BBPBGLIB-246. [Werner
  Van Geit]
- Only serialize sections when really necessary. [Werner Van Geit]
- Disable bgscratch tests until soma-connection issue is resolved
  (import3d changes connect soma at different point to dendrites,
  changes results) [Werner Van Geit]
- Updated two cell test sims to reflect import3d change in neurodamus.
  [Werner Van Geit]
- Fixed bluepy deprecation warnings. [Werner Van Geit]
- Fixed pep8 warning. [Werner Van Geit]


2.5 (2015-10-28)
----------------
- Updated to use the new BlueConfig parsing. [Mike Gevaert]
- Disable warning in dendrogram.py. [Werner Van Geit]
- Added test for existence of neurodamus dirs. [Werner Van Geit]
- Added 'show figure' switch in add_dendrogram. [Werner Van Geit]
- Improved dendrogram plotting. [Werner Van Geit]
- Ignoring two new hdf5 file introduced in Neurodamus. [Werner Van Geit]
- Updated doc to reflect new repo url. [Werner Van Geit]
- Fix an issue with relative linear stimuli. [Werner Van Geit]
- Small commit to test new repo. [Werner Van Geit]
- Added support RelativeLinear BlueConfig stimulus. [Werner Van Geit]
- Fixed pylint warning in cell.py. [Werner Van Geit]
- Unit tests for pulsestim now working All two circuit simulations have
  been rerun. [Werner Van Geit]
- Merge branch 'sideloadsyn' of ssh://bbpgit.epfl.ch/sim/BGLibPy into
  sideloadsyn. [Werner Van Geit]

  Conflicts:
  	test/test_ssim.py
- Pylint pulse stimp test. [Werner Van Geit]
- Merge branch 'sideloadsyn' of ssh://bbpgit.epfl.ch/sim/BGLibPy into
  sideloadsyn. [Werner Van Geit]
- Added simple test for pulse stimulus. [Giuseppe Chindemi]
- Added partial support for Pulse stimulus, missing Offset handling.
  [Giuseppe Chindemi]
- Added simple test for pulse stimulus. [Giuseppe Chindemi]
- Added partial support for Pulse stimulus, missing Offset handling.
  [Giuseppe Chindemi]
- Pylint pulse stimp test. [Werner Van Geit]
- Added simple test for pulse stimulus. [Giuseppe Chindemi]
- Added partial support for Pulse stimulus, missing Offset handling.
  [Giuseppe Chindemi]
- Recreated simulation results regression tests on two cell circuit for
  on CSCS viz. [Werner Van Geit]
- Made two_cell circuit tests independent of bgscratch Little bit of
  pylinting in test_ssim. [Werner Van Geit]
- Fixed an error in the documentation of intersect_pre_gids. [Werner Van
  Geit]
- Disabled pylint message. [Werner Van Geit]
- Added ability to specify cvode minstep and maxstep to simulation.
  [Werner Van Geit]
- Fixed pylint warning. [Werner Van Geit]
- Added sentence to forwardskip documentation. [Werner Van Geit]
- Added forward_skip_value to simulation and ssim. [Werner Van Geit]
- Added more verbosity. [Werner Van Geit]
- Raise exception if add_replay is used with synapse_detail < 1. [Werner
  Van Geit]
- Added base_noise_seed to ssim constructor. [Werner Van Geit]
- Merge branch 'ttx' [Werner Van Geit]
- Replaced 'pip' with 'python -m pip.__main__' to work around long path
  lengths on CSCS viz. [Werner Van Geit]
- Merge branch 'master' into ttx. [Werner Van Geit]
- Added ttx tests to BGLibPy. [Werner Van Geit]
- Replaced 'pip' with 'python -m pip.__main__' to work around long path
  lengths on CSCS viz. [Werner Van Geit]
- Added show_progress to ssim.run() [Werner Van Geit]
- Fixed pep8 error. [Werner Van Geit]
- Fixed pep8 error. [Werner Van Geit]
- Fixed pylint warnings. [Werner Van Geit]
- Don't call re_init_rng when cell is made passive. [Werner Van Geit]
- Ignore .coverage. [Werner Van Geit]
- Disabled automatic printing of header when importing BGLibPy Added
  function print_header to replace printing of header, can be called by
  user Simulation is no longer checking if t < maxtime, this was a bug.
  [Werner Van Geit]
- Replaced implementation of add_ramp with that of add_stim_ramp.
  [Werner Van Geit]
- Removed dt argument from add_tstim_ramp. [Werner Van Geit]


2.4 (2015-01-21)
----------------
- Added add_voltage_recording / get_voltage_recording. [Werner Van Geit]
- Added add_step method to cell that adds a traditional iclamp. [Werner
  Van Geit]
- Changed behavior of HOC_LIBRARY_PATH. If environment already has a
  HOC_LIBRARY_PATH it will be appended after the BGLibPy
  HOC_LIBRARY_PATH. [Werner Van Geit]
- Made method a static function. [Werner Van Geit]
- Merge branch 'master' of ssh://bbpgit.epfl.ch/sim/BGLibPy. [Werner Van
  Geit]
- Now possible to specify section/segx in add_ramp. [Werner Van Geit]
- Merge branch 'master' of ssh://bbpgit.epfl.ch/sim/BGLibPy. [Werner Van
  Geit]
- Update cell info_dict to caste some strings to integers. [Werner Van
  Geit]
- Remove useless print statement. [Werner Van Geit]
- Removed synutils.inc dependence. [Werner Van Geit]
- Reraise exception if neuron import fails. [Werner Van Geit]
- Merge branch 'master' of ssh://bbpgit.epfl.ch/sim/BGLibPy. [Werner Van
  Geit]
- Fixed small bug introduced by previous commit. [Werner Van Geit]
- Create python connection objects even if no real connection to
  presynaptic cell or replay spiketrain. [Werner Van Geit]
- Now we raise original exception when bluepy import fails. [Werner Van
  Geit]
- Fixed apical trunk function, it added apic[0] twice. [Werner Van Geit]
- Disable cvode for holding_current. [Werner Van Geit]
- Added tools.holding_current function. [Werner Van Geit]
- Fixed an issue in grindaway because an integer division instead of a
  float division. [Werner Van Geit]
- Applied a fix to euclid_section_distance. [Werner Van Geit]
- Added function to find the euclidian distance between two sections in
  a morphology. [Werner Van Geit]
- Fixed small bug in apical trunk calculation function. [Werner Van
  Geit]
- Merge branch 'master' of ssh://bbpgit.epfl.ch/sim/BGLibPy. [Werner Van
  Geit]
- Added more documentation to ssim. [Werner Van Geit]
- Disabled load_nrnmech test, because its not working yet. [Werner Van
  Geit]
- Added ability to enable cvode in ssim Added ability to specify seed in
  ssim. [Werner Van Geit]
- Pushing soma when creating cell, adding time recording requires a
  section to have been pushed. [Werner Van Geit]
- Moved test python files to binary directory before running tests.
  [Werner Van Geit]
- Merge branch 'master' of ssh://bbpgit.epfl.ch/sim/BGLibPy. [Werner Van
  Geit]
- Add synapses even when there is no connection block in the BlueConfig
  Show syn_type in info_dict of synapse. [Werner Van Geit]
- Added some verbosity. [Werner Van Geit]
- Made ENABLE_PIP=OFF work correctly. [Werner Van Geit]
- Added version to bglibpy python package. [Werner Van Geit]
- Disabled I0012 in pylint. [Werner Van Geit]


2.2 (2014-07-17)
----------------
- Fixed pylint / pep8 after setup.py introduction. [Werner Van Geit]
- Made setup.py changes run on lviz. [Werner Van Geit]
- Tests run after setup.py changes. [Werner Van Geit]
- First installation using setup.py works. [Werner Van Geit]
- Started with making bglibpy pip installable. [Werner Van Geit]
- Added switches to cmake scripts to disable coverage / xunits. [Werner
  Van Geit]
- Made sure right bluepy gets picked up by pylint. [Werner Van Geit]
- Added restriction of coverage to bglibpy. [Werner Van Geit]
- Cleaned up runtests.sh.in. [Werner Van Geit]
- Updated runtests to ignore .coverage. [Werner Van Geit]
- Added xunit and coverage output. [Werner Van Geit]
- Fixed pep8 warning in cell.py. [Werner Van Geit]
- Added pep8 target, introduced pep8 error on purpose in cell.py.
  [Werner Van Geit]
- All pylint warnings are solved. [Werner Van Geit]
- Solved pylint warnings in psection and simulation. [Werner Van Geit]
- Fixed pylint issues. Also solved an error introduced in previous
  commit. [Werner Van Geit]
- Solved pylint errors ssim. [Werner Van Geit]
- Solved more pylint issues. [Werner Van Geit]
- Solved some pylint errors. [Werner Van Geit]
- Disabled I0011 (prevents locally disabling warnings) in pylint.
  [Werner Van Geit]
- Added pylint target. [Werner Van Geit]
- Merge branch 'master' of ssh://bbpgit.epfl.ch/sim/BGLibPy. [Werner Van
  Geit]

  Conflicts:
  	src/cell.py
- Disabled 'use of eval' pylint warning. [Werner Van Geit]
- Merge branch 'master' of ssh://bbpgit.epfl.ch/sim/BGLibPy. [Werner Van
  Geit]
- Made small change to let Jenkins push the changes. [Werner Van Geit]
- Updated build.sh.lviz.example. [Werner Van Geit]
- Pylint fix in cell.py. [Werner Van Geit]
- Added info_dict() to Cell, Synapse and Connection. [Werner Van Geit]
- Small cleanup in cell.py. [Werner Van Geit]
- Merge branch 'master' of ssh://bbpgit.epfl.ch/sim/BGLibPy. [Werner Van
  Geit]
- Added a comment to src/cell.py. [Werner Van Geit]
- Pylinting. [Werner Van Geit]
- Raise exception when encountering stimulus that is not supported.
  [Werner Van Geit]
- Fixed some pylint warnings. [Werner Van Geit]
- Disabled some pylint warnings. [Werner Van Geit]
- Fixed pep8 error in cell.py. [Werner Van Geit]
- Fixed code to read site-packages dir in case a virtualenv print "using
  ..." messages when starting python. [Werner Van Geit]
- Moved creation of current_version.txt. [Werner Van Geit]
- Fixed 'too many arguments' error in doc upload. [Werner Van Geit]
- Documentation uploading is now done by a shell script. [Werner Van
  Geit]
- Added hbpcol build example. [Werner Van Geit]
- Removed install location module file. [Werner Van Geit]
- Removed adding cmake output files from documentation upload. [Werner
  Van Geit]
- Fixed a bug so that index.html gets upload to the bbp documentation.
  [Werner Van Geit]
- Changed order so to git add in doc_upload adds all files including
  index.html. [Werner Van Geit]
- Fixed a doc_upload dependencies issue. [Werner Van Geit]
- Disabled upload of dirty source directories. [Werner Van Geit]
- Put git push in dry-run mode. [Werner Van Geit]
- Define BGLIBPY_MAINVERSION in CMake. [Werner Van Geit]


2.1 (2014-04-07)
----------------
- Updated documentation repo to point to bbpcode. [Werner Van Geit]
- Changed commit message for doc build. [Werner Van Geit]
- Added doc upload to BBP documentation server, still need to activate
  actual push. [Werner Van Geit]
- Update Lausanne viz build example script. [Werner Van Geit]
- Added version check of neuron to disable/enable renaming templates.
  [Werner Van Geit]
- Merge branch 'master' into samenametemplate. [Werner Van Geit]
- Removed CMake/oss directory. [Werner Van Geit]
- Merge branch 'master' into samenametemplate. [Werner Van Geit]
- Merge branch 'master' of ssh://bbpgit.epfl.ch/sim/BGLibPy. [Werner Van
  Geit]
- Added lbgq build script. [Werner Van Geit]
- Enabled repeating template fix. [Werner Van Geit]
- Started adding code to rename a template in case a template with the
  same was already loaded before. Disabled final functionality because
  neuron crashes when loading a template using HocObject. [Werner Van
  Geit]
- Merge branch 'master' of ssh://bbpgit.epfl.ch/sim/BGLibPy. [Werner Van
  Geit]
- Merge branch 'master' of ssh://bbpgit.epfl.ch/sim/BGLibPy. [Werner Van
  Geit]
- Updated installation instructions to point to new bbpcode repo of
  Neurodamus. [Werner Van Geit]
- Fixed small syntax warning in CMakeLists.txt. [Werner Van Geit]
- Increase timeout on multiprocessing call, Jenkins plan was sometimes
  failing because it was too slow. [Werner Van Geit]
- Updated documentation to reflect the location change of the BluePy
  repository (-> Gerrit) [Werner Van Geit]
- Merge branch 'master' of ssh://bbpgit.epfl.ch/sim/BGLibPy. [Werner Van
  Geit]
- Removed parse error of runtests.sh.in on Ubuntu 13.10. [Werner Van
  Geit]
- Updated installation documentation to reflect the new location of the
  BluePy setup.py. [Werner Van Geit]
- Merge branch 'master' of ssh://bbpgit.epfl.ch/sim/BGLibPy. [Werner Van
  Geit]
- Removed a double installation of tools.py. [Werner Van Geit]
- Disabled xunit output of nosetests, since the ancient version of
  nosetests on the Jenkin build nodes / Viz cluster doesn't support
  this. [Werner Van Geit]
- Added junit output of nosetests. [Werner Van Geit]
- Commented out nose attribute selector code, since this is plugin is
  not available on our test machines with an ancient OS. [Werner Van
  Geit]
- Merge branch 'master' of ssh://bbpgit.epfl.ch/sim/BGLibPy. [Werner Van
  Geit]
- Let CMake print the hostname to stdout. [Werner Van Geit]
- Added capability to disable unit tests that require bgscratch Small
  fix in pre_gid search. [Werner Van Geit]
- Print the neuron installation path from cmake Added an example build
  script for bglibpy on the Lugano viz cluster. [Werner Van Geit]
- Added functionality to get the gids of the presynaptic cells of a
  cell. [Werner Van Geit]
- Add common CMake files. [Werner Van Geit]
- Added BBPSaucy to CMakelists. [Werner Van Geit]
- Expanded the comment of the SSim constructor. [Werner Van Geit]
- Merge branch 'master' of ssh://bbpgit.epfl.ch/sim/BGLibPy. [Werner Van
  Geit]
- Shortened one line. [Werner Van Geit]
- Cleaned up code. [Werner Van Geit]
- Cleaned up code. [Werner Van Geit]
- Cleaned up psection.py. [Werner Van Geit]
- Prevented loading of out.dat if add_replay=True is not specified.
  [Werner Van Geit]
- Merge branch 'master' of ssh://bbpgit.epfl.ch/sim/BGLibPy. [Werner Van
  Geit]
- Cleaned up comments in cell.py. [Werner Van Geit]
- Fixed an issue for user for which the neuron binaries are install in
  $PREFIX/bin instead of $PREFIX/$ARCH/bin. [Werner Van Geit]
- Merge branch 'master' of ssh://bbpgit.epfl.ch/sim/BGLibPy. [Werner Van
  Geit]
- Fixing doc in cell.py to comply PEP257. [Werner Van Geit]
- Cleaned up code. [Werner Van Geit]
- Cleaned up the SSim code. [Werner Van Geit]
- Cleaned up the code. [Werner Van Geit]
- Merge branch 'master' of ssh://bbpgit.epfl.ch/sim/BGLibPy. [Werner Van
  Geit]
- Changed path of nrnpython on santiago test machine. [Werner Van Geit]
- Fixed segment.x in cell.py. [Werner Van Geit]
- Cleaned up Simulation progress bar. [Werner Van Geit]
- Improved the progress bar. [Werner Van Geit]
- Added progress bar to Simulation. [Werner Van Geit]
- Added area calculation to cell.py. [Werner Van Geit]
- Fixed small bug in dendrogram. [Werner Van Geit]
- Added functions that return the release morphologies and ccelss
  directories. [Werner Van Geit]
- Brought cell.py to comply to pep8 standard. [Werner Van Geit]
- Added a function to cell to make a neuron passive. [Werner Van Geit]
- Implemented ForwardSkip in BGLibPy and added a unit test for it.
  [Werner Van Geit]
- Added ssim support for replay to bonus projection synapses, with
  example.  Does not parse BlueConfig yet for BonusSynapseFile params,
  because this syntax is about to change in bglib to support multiple
  projections. [Eilif Muller]
- Merge remote branch 'origin/master' into ebmuller. [Eilif Muller]
- Connection blocks with dest or src targets that don't exist are now
  ignored. [Werner Van Geit]
- Using numpy.testing.assert_array_almost_equal to compare arrays for
  tapering test. [Werner Van Geit]
- Replaced assert_equal with assert_almost_equal for tapering test.
  [Werner Van Geit]
- Added a test for tapering when using delete_axon with arguments in
  BGLib. [Werner Van Geit]
- Fixing teardown in SSim test suite. [Werner Van Geit]
- Added the properties syns and hsynapses back to the cell object.
  [Werner Van Geit]
- Changed if statement for pre_cell and pre_spiketrain in Connection, so
  that it can handle generators as spiketrains. [Werner Van Geit]
- Merge branch 'master' of ssh://bbpgit.epfl.ch/sim/BGLibPy. [Werner Van
  Geit]
- Changed api.rst, so that source links are shown again in the
  documentation. [Werner Van Geit]
- Renamed Bluebrain to bbp. [Werner Van Geit]
- Added functions to synapse to check if the synapse is inhibitory or
  excitatory. [Werner Van Geit]
- Added new functionality in instantiate_gids to independendly
  enable/disable noise and hyperpolarizing stimuli. [Werner Van Geit]
- Added build dir to .gitignore. [Werner Van Geit]
- Updated README. [Werner Van Geit]
- Removed some useless comments. [Werner Van Geit]
- Finished added an internal representation for section. [Werner Van
  Geit]
- Starting to create an internal BGLibPy structure of a cell with
  psections and psegments. [Werner Van Geit]
- Removed architecture reference from module help. [Werner Van Geit]
- Added support for environment modules. [Werner Van Geit]
- Remove showdenddiam function because it's deprecated. [Werner Van
  Geit]
- Added r in front of regular expression string. [Werner Van Geit]
- Merge branch 'master' of ssh://bbpgit.epfl.ch/sim/BGLibPy. [Werner Van
  Geit]
- Updated doc of bglibpy.tools.search_hyp_current_replay_gidlist.
  [Werner Van Geit]
- Added date to button of doc pages. [Werner Van Geit]
- Merge branch 'ebmuller' [Werner Van Geit]
- Updated the documentation of a set of functions. [Werner Van Geit]
- Removed namespace polution in SSim. [Werner Van Geit]
- Cleanup. [Werner Van Geit]
- Got Ben's unit tests for get_gids_of_mtypes() running. [Werner Van
  Geit]
- Fixed problems in Ben's unit tests because pickled files were not
  saved in the repo. [Werner Van Geit]
- Fixed an bug after renaming get_section to get_hsection. [Werner Van
  Geit]
- Merge branch 'btn' [Werner Van Geit]

  Conflicts:
  	src/ssim.py
  	src/tools.py
- Ssim.get_gids_of_mtypes + tests. [Benjamin Torben-Nielsen]
- Added get_gitd_of_mtypes helper to ssim; uses the self.bc_simulation
  to handle queries. TODO: add test. [Ben Torben-Nielsen]
- Moved get_gid_of_mtypes froom tools.py, to be moved to ssim. [Ben
  Torben-Nielsen]
- Added automatic deprecation doc to deprecated function Extended
  documentation of some cell functions. [Werner Van Geit]
- Added a haiku-bbp theme, to fix an issue with haiku and numpydoc
  interaction. [Werner Van Geit]
- Added a ~ to links in the tutorial to shorten the linked name. [Werner
  Van Geit]
- Replace ::code with ::code-block in rst files. [Werner Van Geit]
- Added pre_gid field to Synapse class. [Werner Van Geit]
- Added some example to the tutorial. [Werner Van Geit]
- Documentation now works with numpydoc. [Werner Van Geit]
- Documentation now generates autosummary for all the modules correctly.
  [Werner Van Geit]
- Fixed a erroneous move of index.rst to introduction.rst. [Werner Van
  Geit]
- Fixed Paramters to Parameters in ssim doc. [Werner Van Geit]
- Extended the documentation, and reordered things a bit. [Werner Van
  Geit]
- Enabling numpydoc again. [Werner Van Geit]
- Disabled numpydoc temporarily until it works in the bamboo plans.
  [Werner Van Geit]
- Added support for BBPQUANTAL in the CMakeLists.txt. [Werner Van Geit]
- Removed checks in instantiate_gids to see if no illegal combinations
  of options are given, it clashes with the synapse_detail setting.
  [Werner Van Geit]
- Search_hyp_current_replay_imap: support to override cpu_count, other
  minor fix. [Eilif Muller]
- Search_hyp_current_replay: Making return values for non-convergence
  conformant to layout for successful cases to avoid complex downstream
  logic. [Eilif Muller]
- Merge remote branch 'origin/master' into ebmuller. [Eilif Muller]
- Merge remote branch 'origin/master' into ebmuller. [Eilif Muller]
- Merge remote branch 'origin/master' into ebmuller. [Eilif Muller]
- Merge remote branch 'origin/master' into ebmuller. [Eilif Muller]


2.0 (2013-04-02)
----------------
- Updated version to 2.0. [Werner Van Geit]
- Updated the documentation string of instantiate_gids to reflect the
  multi-cell changes Fixed a bug in Connection concerning the variable
  name of the netcon added an example for a multicell replay. [Werner
  Van Geit]
- Finished implementation of multi cell functionality of BGLibPy
  Connection now correctly sets the weight of the real connections Added
  unit test for real connections. [Werner Van Geit]
- Trying to get connect2target working, waiting for response from
  M.Hines. [Werner Van Geit]
- Implemented connections between multiple cells, but it still core
  dumps. [Werner Van Geit]
- Added a new synapse class. Still in an inconsistent state before
  multicell works. [Werner Van Geit]
- Large rewrite of ssim to make it more readable. Separate functions to
  add the stimuli, synapses, cells etc. This code is not finished, and
  will not function correctly. [Werner Van Geit]
- Preparing to make it possible to connect several cells in a network: -
  created a Connection class that represents a network connection in
  BGLibPy. [Werner Van Geit]
- Renamed some variables in ssim to make them more readable only parse
  out.dat once. [Werner Van Geit]
- Moved installation guide into separate file. [Werner Van Geit]
- Enforced CMake 2.8, since we're not testing for CMake 2.6. [Werner Van
  Geit]
- Added two simple examples of BGLibPy usecases. [Werner Van Geit]
- Solved an issue in CMakeLists.txt in which some interference with
  apparently BuildYard or something, make the configure_file to write
  the paths.config in the wrong directroy. [Werner Van Geit]
- Starting with installation tutorial. [Werner Van Geit]
- Added other modules to documentation conf.py for the doc now get's the
  right location of BGLibPY. [Werner Van Geit]
- Starting doc making in CMakeLists.txt. [Werner Van Geit]
- Merge branch 'imap_parallel' [Werner Van Geit]

  Conflicts:
  	src/tools.py
- Search_hyp_current_replay_imap now internally uses asynchronous
  parallelization. It returns a generator, so that the user can, one by
  one retreive the asynchronous results. [Werner Van Geit]
- Added imap function to calculate hypvoltage. [Werner Van Geit]
- Merge branch 'btn' [Werner Van Geit]
- Cleaned up the doc directory. TODO: resolve issue with autosummary in
  api.rst. [Ben Torben-Nielsen]
- First: commit, second: clean up the doc mess. [Ben Torben-Nielsen]
- Too much documentation. [Ben Torben-Nielsen]
- Merge remote-tracking branch 'origin/master' into btn. [Ben Torben-
  Nielsen]
- Werner revised the intersect_pre_gid for loop. [Ben Torben-Nielsen]
- Fixed a bug in tools.py where the same variable full_voltage was
  erroneously used twice. [Werner Van Geit]
- Changed the behavior of search_hyp_current_replay_gidlist so that it
  implements a timeout in case one of the subpool workers doesn't return
  in time. [Werner Van Geit]
- Merge branch 'ebmuller' [Werner Van Geit]
- Merge remote branch 'origin/master' into ebmuller. [Eilif Muller]
- Minor fixes: consistency of return values for return_fullrange modes,
  multiprocessing map uses cpu count, additional doc clarifications.
  [Eilif Muller]
- Merge remote branch 'origin/master' into ebmuller. [Eilif Muller]
- Minor fixes: consistency of return values for return_fullrange modes,
  multiprocessing map uses cpu count, additional doc clarifications.
  [Eilif Muller]
- Added code to the delete() function of cells, so that they destroy the
  circular dependencies introduced by FInitializeHandler SSim will now
  call this delete() function on all its cells during destruction.
  [Werner Van Geit]
- Add support for the 'delay' field of a connection block in a
  BlueConfig. [Werner Van Geit]
- Hardened the SSim connection block reader against ignoring any
  unsupported fields in these block. [Werner Van Geit]
- Merge branch 'ebmuller' [Werner Van Geit]
- Merge remote branch 'origin/master' into ebmuller. [Eilif Muller]
- Added option to check for spiking (and if so, return None) for
  calculate_SS_voltage_subprocess.  Default behaviour unchanged. [Eilif
  Muller]
- Added methods to reset synapse state. [Eilif Muller]
- Merge remote branch 'origin/master' into ebmuller. [Eilif Muller]
- Merge remote branch 'origin/master' into ebmuller. [Eilif Muller]
- Added sections keyword to execute_neuronconfigure method. [Eilif
  Muller]
- Merge remote branch 'origin/master' into ebmuller. [Eilif Muller]
- Merge remote branch 'origin/master' into ebmuller. [Eilif Muller]
- Merge with origin/master. [Eilif Muller]
- Added failure status for add_replay_synapse, instantiate_gids now has
  a synapse_detail=0 option. [Eilif Muller]
- Made default edgecolor of psegment 'black' [Werner Van Geit]
- Removed finitialize from constructor of dendrogram. [Werner Van Geit]
- Made a warning in runtest.sh more visible. [Werner Van Geit]
- Removed all reference in other modules to getTime and getSomaVoltage.
  [Werner Van Geit]
- Removed all references to addRamp in other modules. [Werner Van Geit]
- Dendrogram is working again Refactored some functions in cell.py.
  [Werner Van Geit]
- Reenabled to ability to add live plots. This time the code is using
  cvode.event callback function, so that it doesn't interfere with the
  time step of the simulation. [Werner Van Geit]
- Renamed function that parses the out.dat in ssim Created a unit test
  for this function Added script that runs coverage analysis on the unit
  tests. [Werner Van Geit]
- Added a warning to runtests.sh to warn users to rebuild BGLibPy before
  executing runtests.sh. [Werner Van Geit]
- Added a unit test for search_hyp_current_replay_gidlist Slight changed
  the API of search_hyp_current_replay_gidlist, so that it also returns
  the time trace, in addition to the voltage trace. [Werner Van Geit]
- Updated the BlueConfigs in the unit tests to reflect the changes in
  bgscratch directory structure on BG/Q. [Werner Van Geit]
- Adding kwargs to search_hyp_current_replay_gidlist, instead of a
  specifying an entire list of kwargs that have to percolate down.
  [Werner Van Geit]
- Disable show_progress by default in the run() of Simulation. [Werner
  Van Geit]
- Made it possible to specify the test as an argument to runtests.sh.
  [Werner Van Geit]
- Small cleanup of comments in test_ssim. [Werner Van Geit]
- Added the ability to show the progress of a simulation to the run()
  function of Simulation. [Werner Van Geit]
- Calculate_SS_voltage_replay_subprocess now returns a voltage of a
  'full time range' of the simulation after it is done, not just the
  time between start_time / stop_time. [Werner Van Geit]
- Added function documentation to search_hyp_current_replay_gidlist.
  [Werner Van Geit]
- Merge branch 'master' of ssh://bbpgit.epfl.ch/sim/BGLibPy. [Werner Van
  Geit]
- Add --tags to the git describe, so that we don't depend on annotated
  tags. [Werner Van Geit]
- Changed the verbose level of some messages in ssim to level 2. [Werner
  Van Geit]
- Added a function to tools.py called search_hyp_current_replay_gidlist
  It search for a list of gids, the current injection amplitude
  necessary to bring the cells to a target voltage. [Werner Van Geit]
- Added CMake code that checks for the version of Neuron and BGLib used
  during compilation. The versions can be accessed by the variable
  build_versions of the module. [Werner Van Geit]
- Added __version__, version and VERSION variables to the module that
  contain the git-repository version of BGLibPy. [Werner Van Geit]
- Dummy commit, trying out versioning. [Werner Van Geit]


1.0 (2013-03-07)
----------------
- Werner revised the intersect_pre_gid for loop. [Ben Torben-Nielsen]
- Merge branch 'master' of ssh://bbpgit.epfl.ch/sim/BGLibPy. [Werner Van
  Geit]
- Merge branch 'master' of ssh://bbpgit.epfl.ch/sim/BGLibPy. [Werner Van
  Geit]
- Added an option intersect_pre_gids to control from which pre_gids
  synapses are generated in instantiate_gids of SSim. [Werner Van Geit]
- Added pylint ignore in cell.py. [Werner Van Geit]
- Create_sims_twocell.py now uses a pybinreports installation, instead
  of a magic soma2h5.py file somewhere. [Werner Van Geit]
- Added a version of the test circuit nrn.h5 that has track times
  disabled. [Werner Van Geit]
- Testing if disabling track times in h5py works. [Werner Van Geit]
- Added a warning when a spontminis statement in a BlueConfig is ignored
  because it's preceded by another one. [Werner Van Geit]
- Added an extra unit test to the SynapseID test, to see if the
  BlueConfig 'with' SynapseID generates a different result than the one
  without it. [Werner Van Geit]
- Added unit test for SynapseID functionality of BGLib Fixed some issues
  in the implementation of the SynapseID Replicated a 'feature' of BGLib
  where only the first Connection block sets SpontMinis. [Werner Van
  Geit]
- Added functionality that handles the SynapseID field in Connection
  blocks. [Werner Van Geit]
- Made runtests.sh fail if one of both tests fail. [Werner Van Geit]
- Checkout for directory of loading in test_load.py instead of
  __init__.py. [Werner Van Geit]
- Added a test to see if the module is loaded from the right path.
  [Werner Van Geit]

  Removed hardcoded path in tests to /home/vangeit
- Add sim_twocell_neuronconfigure. [Werner Van Geit]
- Made all the class inherit from 'object' [Werner Van Geit]
- Added an exception in case the Cell template was not found. [Werner
  Van Geit]
- Deprecated addCell in favor of add_cell Removed print statement in
  cell.py. [Werner Van Geit]
- Added a BlueConfig template to test the two cell simulation with
  NeuronConfigure. [Werner Van Geit]
- Enabled all the tests again, was only running test_ssim. [Werner Van
  Geit]
- Added support for '%g' in NeuronConfigure block. [Werner Van Geit]
- Added the ability to parse NeuronConfigure BlueConfig blocks to ssim.
  [Werner Van Geit]
- Removed test_ssim selection from nosetest in runtests.sh.in. [Werner
  Van Geit]
- Added ballstick.asc and hoc to ballstick_test directory, otherwise the
  bglib simulatino there doesn't run. [Werner Van Geit]
- Changed the default value of 'distance' in synlocation_to_segx to 0.5,
  the synchronize with BGLib. Before the Chand-AIS bug was fixed in
  BGLib the default value was -1. [Werner Van Geit]

  Changed the circuit for the unit tests of SSim to a newer version, that ran with a version of BGLib with the Chand-AIS bug
- Added an extra warning in case cvode was activated outside of
  Simulation, to warn that this might prevent templates with stochastic
  channels to load. [Werner Van Geit]
- Changes concerning the behavior of cvode=True in Simulation.run(). The
  function will now save the old state of cvode, will set the state of
  cvode to 'cvode' argument of the function, will then run the
  simulation, and will afterwards put the state back This change was
  necessary to allow the loading of template with stochastic channels,
  after running of simulation with cvode=True. [Werner Van Geit]
- Added a unit test for calculate_SS_voltage. [Werner Van Geit]
- Added functionality to tools.calculate_SS_voltage_subprocess to check
  if a template contains a stochastic channel, now it will automatically
  disable cvode if that's the case. [Werner Van Geit]
- Changed the way the circuitpath is set for the twocell circuit
  example, so that it's not hardcoded to /home/vangeit. [Werner Van
  Geit]
- Less calls to an improved parse_and_store..., part II. [Ben Torben-
  Nielsen]
- Less calls to an improved parse_and_store... [Ben Torben-Nielsen]
- Created external_tools dir with tools used by the tests, ideally this
  directory should not exist, but this is a temporary place to save
  tools that don't have a real home somewhere else. [Werner Van Geit]
- Added test to see if dimensions of the ballstick load correctly.
  [Werner Van Geit]
- Commented out path to green function python file on viz cluster.
  [Werner Van Geit]
- Ballstick is now part of the unit test suite. [Werner Van Geit]
- Added a check in the unit tests to see if the diameters / lengths of
  soma,basal and apical are loaded correctly. [Werner Van Geit]
- Regenerated examples. [Werner Van Geit]
- Working version of ballstick, no analytic solution comparison yet.
  [Werner Van Geit]
- Merge branch 'master' of ssh://bbpgit.epfl.ch/sim/BGLibPy. [Werner Van
  Geit]
- Working on ballstick unit test, unfinished, temporarily disabled test.
  [Werner Van Geit]
- Added a unit test that tests a two_cell simulation with replay, minis
  and stimuli. [Werner Van Geit]
- Added a README for twocell_circuit. [Werner Van Geit]
- Syntactic changes in the out.dat parser in SSim In replay unit test,
  now add dummy spike because BGLib cannot handle an empty out.dat.
  [Werner Van Geit]
- Added unit tests for two cell circuit with minis. [Werner Van Geit]
- Cleaned up the output to stdout. [Werner Van Geit]
- Merge branch 'master' of ssh://bbpgit.epfl.ch/sim/BGLibPy. [Werner Van
  Geit]
- Move synapseconfigure block in add_replay_synapse to a place after
  setting the Use and Dep etc, otherwise the values get overwritten.
  [Werner Van Geit]
- Added finitialize to the initialization of a Cell. Solved a bug in
  which the diameters of the morphologies were not set correctly
  WARNING: this change will mess up replays when more than one cell is
  loaded. [Werner Van Geit]
- Merge remote-tracking branch 'origin/merge-vangeit' [Werner Van Geit]
- Small change in README. [Werner Van Geit]
- Added a unit test for the two cell circuit ssim with replay. [Werner
  Van Geit]
- Changed instantiate_gids call to allow more specific control on which
  level mechanism are loaded from the large simulation. [Werner Van
  Geit]
- Added noisestim unit test to ssim. [Werner Van Geit]
- Updating the naming of sim_twocell. [Werner Van Geit]
- Fixed small bug where print was still used in ssim. [Werner Van Geit]
- Fixed syntactic error in test_ssim. [Werner Van Geit]
- Added two files that were missing from the previous commit. [Werner
  Van Geit]
- First unit test that compares ssim with real bglib now working.
  [Werner Van Geit]
- SSim now uses printv / printv_err to print messages based on verbose
  level. [Werner Van Geit]
- Fixed bug in run of ssim, tstop and dt should be cast to a float when
  reading from the BlueConfig. [Werner Van Geit]
- SSim run now default to the tstop and dt from the BlueConfig. [Werner
  Van Geit]
- Added a verbose level function. Use printv(message, verbose_level) to
  print depending on the verbose level. [Werner Van Geit]
- Fixing the script to create twocell_empty unit test sim. [Werner Van
  Geit]
- Added unit test for deprecation warning. [Werner Van Geit]
- Merge branch 'ebmuller' [Werner Van Geit]
- Fix to the deprecation decorator to support python 2.6. [Eilif Muller]
- Moved example files for unit tests to 'example' directory Started
  building a script to create a test simulation. [Werner Van Geit]
- Brought the test_ssimm into nosetest format. [Werner Van Geit]
- Moved more scripts to create_extracted. [Werner Van Geit]
- Changes to scripts to test extracting circuits. [Werner Van Geit]
- Add script to make test circuit. [Werner Van Geit]
- Added test circuit with two cells. [Werner Van Geit]
- Syntactic changes to test_cell. [Werner Van Geit]
- Read BaseSeed instead of baseSeed from BlueConfig Works now if
  BlueConfig contains SynapseReplay (just ignores it) [Werner Van Geit]
- Added support for steps_per_ms run() [Werner Van Geit]
- Removed again the 'epsilon' trick with the dt proposed by M. Hines,
  since this trick is not used in BGLib. [Werner Van Geit]
- Changes in my testextractor script. Preparing to move everything to
  unittest dir. [Werner Van Geit]
- Updates to the testextractor. [Werner Van Geit]
- Renamed the function simulate() to run() in ssim. [Werner Van Geit]
- First working version of testextractor. [Werner Van Geit]
- Added a check to only add synapses to a cell when there is at least
  one presynaptic cell The BaseSeed gets now correctly parsed to an int
  from an integer after it's read from the BlueConfig. [Werner Van Geit]
- Added checks to see if out.dat exists, and if a gid exists when it's
  instantiated. [Werner Van Geit]
- Added a script to test the bluepy extractor, and run a small circuit
  with BGLibPy. [Werner Van Geit]
- Merge branch 'master' of ssh://bbpgit.epfl.ch/sim/BGLibPy. [Werner Van
  Geit]
- Merge branch 'master' of ssh://bbpgit.epfl.ch/sim/BGLibPy. [Werner Van
  Geit]
- Commented out numpy in my testreplay.py. [Werner Van Geit]
- Moved werner tests in separate directories Added a message that shows
  where BluePy is loaded from. [Werner Van Geit]
- Added comments to explain some unit tests. [Werner Van Geit]
- Nosetests now stop after first error. [Werner Van Geit]
- Merge branch 'ebmuller' [Werner Van Geit]
- Changes to use bluepy circuit extractor.  Not yet tested because
  blocked by a bglib module bug on viz cluster. [Eilif Muller]
- Small changes to my own replay tests. [Werner Van Geit]
- Merge branch 'master' of ssh://bbpgit.epfl.ch/sim/BGLibPy. [Werner Van
  Geit]
- Moved import of matplotlib into the appropriate function call. [Werner
  Van Geit]
- Added a flag DBBPSANTIAGO=ON to define the location of nrn on the BBP
  Redhat Santiago test machine. [Werner Van Geit]
- Added BBPQUANTAL as configure option in cmake. [Werner Van Geit]
- Added some extra tests for the Cell class. [Werner Van Geit]
- Changed a call to addRecording to add_recording. [Werner Van Geit]
- Added some comment in the cell.py code. [Werner Van Geit]
- Added some verbose messages. [Werner Van Geit]
- Commented out a debug message that showed the seeds used for the
  minis. [Werner Van Geit]
- Updated my personal test scripts. Changes made to test full replays of
  BGLib. [Werner Van Geit]
- Changed the way the Simulation object runs a simulation. This is now
  done by calling neuron.h.run() for the full period of time. This is at
  the moment the only way to get a near perfect replay of the original
  BGLIB. Breaks all code that depends on python stepping out of Neuron
  every timestep (like live plotting) [Werner Van Geit]
- Merge branch 'master' of ssh://bbpgit.epfl.ch/sim/BGLibPy. [Werner Van
  Geit]
- Revert "Publishing updated achievement" [Werner Van Geit]

  This reverts commit e78d5aa8dda1e9a00cdba0e4a91afd5b7105cf0b.
- Publishing updated achievement. [Werner Van Geit]
- Added a shebang to the shell scripts. [Werner Van Geit]
- Added headers to all the python files. [Werner Van Geit]
- Started adding documentation. [Werner Van Geit]
- Merge branch 'master' of ssh://bbpgit.epfl.ch/sim/BGLibPy. [Werner Van
  Geit]
- Solved a bug in which paths.config was not closed after opening.
  [Werner Van Geit]
- Removed warning when no presynaptic spikes. [Werner Van Geit]
- Prevented crash when no SynpaseConfigure block was present More
  verbose when adding minis. [Werner Van Geit]
- Small syntactic change in reading out.dat. [Werner Van Geit]
- Update way blueconfig file is load in the Pure BGLib test script.
  [Werner Van Geit]
- Fixed some calls to old deprecated functions in cell and plotwindow.
  [Werner Van Geit]
- Ignoring coverage reports in git. [Werner Van Geit]
- Renamed test dir test_cell to cell_example1, because it confused
  nosetests. [Werner Van Geit]
- Importer now load SerializedSections instead of SerializedCell, this
  is now an official file in BlueBrain. [Werner Van Geit]
- Simulation.run uses step again, live updating of plots supported
  again. [Werner Van Geit]
- Merge branch 'master' of ssh://bbpgit.epfl.ch/sim/BGLibPy. [Werner Van
  Geit]
- Temporarily added Eilif's soma2h5.py to my test dir. [Werner Van Geit]
- Fixed some bugs in cell.py: persistent.objects is supposed to be
  replaced with persistent Now code checks if gethypamp and getthreshold
  in a template before assigning the properties. [Werner Van Geit]
- Added example Blueconfig to run BGLib as temporary test. [Werner Van
  Geit]
- Create get_time and get_soma_voltage, deprecated old version Fixed a
  bug where get_target was called on circuit instead of simulation.
  [Werner Van Geit]
- Changed the way the 'run' function works, it now gives complete
  control to neuron until tstop Live plotting WON'T work anymore for the
  time being Also wernertests directory with temporary tests. [Werner
  Van Geit]
- Merge branch 'master' of ssh://bbpgit.epfl.ch/sim/BGLibPy. [Werner Van
  Geit]
- Moved charging of synapses into Cell. [Werner Van Geit]
- Fixed bugs in synlocation_to_segx, now almost contains the same code
  as locationToPoint of BGLib. But there is still an discrepancy, in the
  sense that when distance = -1 (when a synapse is tried to be placed on
  the axon), BGLibPy will put the synapse at location 0, while BGLib
  will NOT place the synapse. [Werner Van Geit]
- Renamed syn_description to connection_parameters. [Werner Van Geit]
- Merge branch 'btn' [Werner Van Geit]

  Conflicts:
  	src/cell.py
  	src/ssim.py
- Panic? Maybe it works now...? [Werner Van Geit]
- No real change, just to resolve a conflict while merging with
  3dd85917e52b2f81cdc328bd512bb00b1e282388. [Werner Van Geit]
- Small refactoring of some variables in Cell. [Werner Van Geit]
- Moved the mini creation to cell.py. [Werner Van Geit]
- Moved ssim noisestim in cell Now using TStim for hyamp stimulus.
  [Werner Van Geit]
- Replaced the out.dat reader with a much smaller version. [Werner Van
  Geit]
- Small code fixing, persistent is now object, not class. [Werner Van
  Geit]
- Resolved an import warning in __init__.py. [Werner Van Geit]
- Code cleanup and detailed code checking. [Werner Van Geit]
- Merge branch 'master' of ssh://bbpgit.epfl.ch/sim/BGLibPy. [Werner Van
  Geit]
- Fixed a bug in ssim add_replay_noise. The variance was not divided by
  100, like in tstim.noise( $4, $5, threshold*$2/100, threshold*$3/100 )
  [Werner Van Geit]
- Removed mechanisms from cell. [Werner Van Geit]
- Fixed a bug concerning distance that was not initialize in
  location_to_point. [Werner Van Geit]
- Add import SSim from ssim to __init__.py. [Werner Van Geit]
- Removed some whitespaces. [Werner Van Geit]
- Merge branch 'btn' [Werner Van Geit]

  Conflicts:
  	src/cell.py
- Ssim now able replay as before with atomized function calls.
  _add_replat_stimuli has to be changed to use BluePy in the near
  future. [Ben Torben-Nielsen]
- Nose test for the replay functionality in bglibpy.ssim.SSim. [Ben
  Torben-Nielsen]
- Deprecated some more functions. [Werner Van Geit]
- Ignore .bglib* files. [Werner Van Geit]
- Added ignore for .bglib files. [Werner Van Geit]
- The importer now adds NRNPYTHONPATH to sys.path. [Werner Van Geit]
- Merge branch 'master' of ssh://bbpgit.epfl.ch/sim/BGLibPy. [Werner Van
  Geit]
- Imported pylab only at moments when it's necessary, to be able to run
  the code without a display variable set. [Werner Van Geit]
- Renamed add_synapse to add_replay_synapse in cell.py. [Werner Van
  Geit]
- Added a shell script to execute the test (after install) [Werner Van
  Geit]
- Removed syns from ssim and put it into cell. [Werner Van Geit]
- Removed --processes from nosetests, not supported on vizcluster Added
  -v to nosetests. [Werner Van Geit]
- Merge remote-tracking branch 'origin/ebmuller' into separate_files.
  [Werner Van Geit]

  Conflicts:
  	.gitignore
  	src/bglibpy.py
  	src/ssim.py
  	test/test_ssim.py
- Added validation of Ben's PSP amplitude code against bglib.  Added
  .gitignore. [Eilif Muller]
- Cosmetic changes to test_ssim.py. [Eilif Muller]
- Moved deprecated functions to the back. [Werner Van Geit]
- Changed header of location_to_point. [Werner Van Geit]
- Moved some cell functionality from ssim to cell (add_synapse,
  get_section, ...) [Werner Van Geit]
- Fixed the test_ssim, to work with Ben's new version of ssim. [Werner
  Van Geit]
- Fixed celsius=-34 arg, and lack of use of it in run function. [Eilif
  Muller]
- Changed the ssim, simulation and cell classes, so that they can handle
  templates with stochastic channels The gid is now passed to the cell
  object, and re_init_rng is called that sets the random seed of the
  stochastic channels dependent on the gid of the cell. [Werner Van
  Geit]
- Merge remote-tracking branch 'origin/btn' into separate_files. [Werner
  Van Geit]

  Conflicts:
  	src/ssim.py
- _evaluate_connection_parameters was prohibitively slow due to many
  bluepy...get_target calls. Solved. [Ben Torben-Nielsen]
- Nose tests for the ball-and-stick model. Part I: comparison of B&S
  models with ExpSyn (requires Willems code for some tests) [Ben Torben-
  Nielsen]
- Added import neuron to tools.py, was bug. [Werner Van Geit]
- Removed check for pythonlibs in CMakeLists.txt, not really necessary.
  [Werner Alfons Hilda Van Geit]
- Disabled the progressbar when loading the gids. [Werner Van Geit]
- Added ctest -VV to build.sh.example. [Werner Van Geit]
- Made it possible to run make test to run the nosetests. [Werner Van
  Geit]
- Merge btn and ebmuller in separate files branch. [Werner Van Geit]
- Merge branch 'ebmuller' into separate_files. [Werner Van Geit]
- Forgot to add these files to the last commit. [Eilif Muller]
- Fixed problem with ProbAMPANMDA_EMS (needs gsyn in nS not uS, so
  scaled gsyn by 1000).  Comparisons in btn_bs_nogreen.py now agree to
  within .05 mv.  Added comparison with Ben's ssim psp, and some
  differing dt, code ssim psp infrastructure and bglib agree to a much
  better margin. [Eilif Muller]
- Merge remote branch 'origin/btn' into ebmuller. [Eilif Muller]
- Refresh of soma.h5 from bglib. [Eilif Muller]
- Merge branch 'btn' into separate_files. [Werner Van Geit]
- Current script to compare BGLIB vs. BGLibPy. [Ben Torben-Nielsen]
- Updated soma.h5 voltage trace with nseg=200 change in ballstick.hoc
  template. [Eilif Muller]
- Merge remote-tracking branch 'origin/ebmuller' into btn. [Ben Torben-
  Nielsen]
- Adding ballstick test circuit and sim, and output using bglib. [Eilif
  Muller]
- Merge remote branch 'origin/master' into ebmuller. [Eilif Muller]
- Work in progress on comparison bglibpy / analytic / bglib. [Ben
  Torben-Nielsen]
- Merge branch 'btn' into separate_files. [Werner Van Geit]

  Conflicts:
  	test/cell_test/cell_test.py
  	test/cell_test/test_cell.hoc
  	test/load_test/load_test.py
- Merge remote-tracking branch 'origin/master' into btn. [Ben Torben-
  Nielsen]
- Add a script to convert ballstick.asc to ballstick.h5. [Werner Van
  Geit]
- Added h5 version of ballstick.asc. [Werner Van Geit]
- Update the ballstick morphology so that it doesn't contain an axon.
  [Werner Van Geit]
- Merge branch 'ebmuller' [Werner Van Geit]
- Added ball-and-stick model test. [Werner Van Geit]
- Put the SerializedCell.hoc back, loading TargetManager.hoc instead
  generates a neuron seg fault. [Werner Van Geit]
- Removed dependency from SerializedCell.hoc, TargetManager.hoc gets
  load now instead. [Werner Van Geit]
- Started adding some tests. [Werner Van Geit]

  Conflicts:
  	test/cell_test/cell_test.py
- Added a new proposal for Connection block parsing, and test cases.
  [Eilif Muller]
- Fixed bug, targets are fetched from simulation object (which includes
  start.target and user.target), error is raised if target not found.
  [Eilif Muller]
- Started adding some tests. [Werner Van Geit]
- Merge branch 'master' into separate_files. [Werner Van Geit]
- Added newline to make a line shorter in cmakelists.txt. [Werner Van
  Geit]
- Changed prefix behaviour to use distutils prefix computer. [Eilif
  Muller]
- Made morph path code remove /h5 if present in the blueConfig, fixed a
  typo: basSeed->baseSeed. [Eilif Muller]
- Trying to solve the issue with 'import neuron' [Werner Van Geit]
- Replace PYTHON_BINARY by 'python' when executing python to find python
  install path. [Werner Van Geit]
- Put all the classes in separate files. [Werner Van Geit]
- The CMakeLists now detects the pythonxxx/site-packages directory from
  the python install. [Werner Van Geit]
- Ran pyflakes, pylint, and pep8 on the code. [Werner Van Geit]
- Merge branch 'btn' [Werner Van Geit]
- Some more functionality for SSIm. [Ben Torben-Nielsen]
- Some of the SSIM (unclean) [Ben Torben-Nielsen]
- Start of the Small-number simulator an extension of bglibpy to add
  powerful replay functionality. [Ben Torben-Nielsen]
- Merge branch 'btn' of ssh://bbpgit.epfl.ch/sim/BGLibPy into btn. [ben]

  Conflicts:
  	src/bglibpy.py
- Nothing to report. [ben]
- Merged with master. [ben]
- Some changes to make bglibpy run on Linsrv2. [Werner Van Geit]
- Fixed string in CMakeLists.txt. [Werner Van Geit]
- Merge branch 'btn' into btn-merge Installing bglibpy in subdirectory
  of site-packages. [Werner Van Geit]

  Conflicts:
  	src/bglibpy.py
- Merge remote-tracking branch 'origin/btn' into btn. [Werner Van Geit]
- Final before other repository. [ben]
- Cleaned bglibpy + moved static methods to tools.py. [ben]
- Creating BTN branch. [ben]
- Added bluepy location for CMake. [Werner Van Geit]
- Removed the 'rm' command from the build.sh.example. [Werner Van Geit]
- Changed the CMakefile so that the mod files only compile when they
  have been changed. [Werner Van Geit]
- Finalized the merge, got code into correct style. [Werner Van Geit]
- Merge branch 'btn' of bbplinsrv2:../torben/bglibpy into btn. [Werner
  Van Geit]

  Conflicts:
  	.gitignore
  	modlib/ProbAMPANMDA.mod
  	modlib/ProbGABAA.mod
  	modlib/tmgInhSyn.mod
  	modlib/utility.mod
  	src/bglibpy.py
  	test/test.py
- Chap. [Benjamin Torben-Nielsen]
- Modifications to get Ben started. [Benjamin Torben-Nielsen]
- Cleaned up the code. [Werner Van Geit]
- Changed cmake install, so that you now have to specify the
  NRNPYTHONPATH. [Werner Van Geit]
- Reworked the installation system, now uses cmake. [Werner Van Geit]
- Added function to show the dendrite section number that come out of
  the soma. [Werner Van Geit]
- Added setup.py script, reorganized structure. [Werner Van Geit]
- Added function to find apical trunk. [vangeit]

  git-svn-id: https://bbpteam.epfl.ch/svn/user/vangeit/bglibpy/trunk@4731 3947adc2-bc01-0410-925f-c2a438adfcc0
- Before changing the way synaptic attenuations are calculated (i.e. no
  synapses on apical shaft anymore) [vangeit]

  git-svn-id: https://bbpteam.epfl.ch/svn/user/vangeit/bglibpy/trunk@4411 3947adc2-bc01-0410-925f-c2a438adfcc0
- Removed some obsolete comments. [vangeit]

  git-svn-id: https://bbpteam.epfl.ch/svn/user/vangeit/bglibpy/trunk@4410 3947adc2-bc01-0410-925f-c2a438adfcc0
- Big update of bglibpy, added ability to show different dendrogram,
  moved modlib into bglibpy, calculating synapse atten. [vangeit]

  git-svn-id: https://bbpteam.epfl.ch/svn/user/vangeit/bglibpy/trunk@3924 3947adc2-bc01-0410-925f-c2a438adfcc0
- Added test script. [vangeit]

  git-svn-id: https://bbpteam.epfl.ch/svn/user/vangeit/bglibpy/trunk@3285 3947adc2-bc01-0410-925f-c2a438adfcc0
- Faster figure update. [vangeit]

  git-svn-id: https://bbpteam.epfl.ch/svn/user/vangeit/bglibpy/trunk@3280 3947adc2-bc01-0410-925f-c2a438adfcc0
- Added files. [vangeit]

  git-svn-id: https://bbpteam.epfl.ch/svn/user/vangeit/bglibpy/trunk@3273 3947adc2-bc01-0410-925f-c2a438adfcc0
- Started. [vangeit]

  git-svn-id: https://bbpteam.epfl.ch/svn/user/vangeit/bglibpy/trunk@3272 3947adc2-bc01-0410-925f-c2a438adfcc0

