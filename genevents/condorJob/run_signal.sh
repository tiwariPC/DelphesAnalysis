#!/usr/bin/env bash
# Exit immediately on error and echo every command
set -e
set -x
ulimit -s unlimited

export ROOTSYS=/usr
export PATH=$PATH:$ROOTSYS/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ROOTSYS/lib
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$ROOTSYS/lib

# -----------------------------
# Input arguments
# -----------------------------
JOBID=$1
MCA="$2"
MSA="$3"
ISEED="$4"

# -----------------------------
# voms proxy initialization
# -----------------------------
export X509_USER_PROXY=$5
voms-proxy-info -all
voms-proxy-info -all -file $5

# -----------------------------
# Job info
# -----------------------------
echo "===================================="
echo "JOBID        = ${JOBID}"
echo "MCA          = ${MCA}"
echo "MSA          = ${MSA}"
echo "ISEED        = ${ISEED}"
echo "===================================="

# -----------------------------
# Prepare launch card
# -----------------------------
LAUNCH_CARD="signal_mA${MCA}_ma${MSA}_iseed${ISEED}.madevent"

cp signal.madevent "${LAUNCH_CARD}"

sed -i \
    -e "s|__MCA__|${MCA}|g" \
    -e "s|__MSA__|${MSA}|g" \
    -e "s|__ISEED__|${ISEED}|g" \
    "${LAUNCH_CARD}"
echo "Prepared launch card: ${LAUNCH_CARD}"

# -----------------------------
# Untar production directory
# -----------------------------
echo "# Untar the Production Dir"
tar -xaf "bbdm_2hdma_type1_case1.tar.gz"
echo "Done"

# -----------------------------
# Run MadEvent
# -----------------------------
./bbdm_2hdma_type1_case1/bin/madevent ${LAUNCH_CARD}
wait

ls -ltr bbdm_2hdma_type1_case1/Events
cp bbdm_2hdma_type1_case1/Events/run_01/tag_1_delphes_events.root sig_bbdm_mA${MCA}_ma${MSA}_${ISEED}_delphes_events.root

if [ -e "sig_bbdm_mA${MCA}_ma${MSA}_${ISEED}_delphes_events.root" ]; then
  until xrdcp -f sig_bbdm_mA${MCA}_ma${MSA}_${ISEED}_delphes_events.root root://eoscms.cern.ch//eos/cms/store/group/phys_susy/sus-23-008/DelphesSignalFiles/sig_bbdm_mA${MCA}_ma${MSA}_${ISEED}_delphes_events.root; do
    sleep 60
    echo "Retrying"
  done
fi
exitcode=$?

if [ ! -e "sig_bbdm_mA${MCA}_ma${MSA}_${ISEED}_delphes_events.root" ]; then
  echo "Error: The python script failed, could not create the output file."
fi
exit $exitcode

echo "===================================="
echo "Job finished successfully"
echo "===================================="
