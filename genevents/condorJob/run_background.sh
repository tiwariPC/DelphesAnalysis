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
BKG_NAME="$2"
ISEED="$3"

# -----------------------------
# voms proxy initialization
# -----------------------------
export X509_USER_PROXY=$4
voms-proxy-info -all
voms-proxy-info -all -file $4

# -----------------------------
# Job info
# -----------------------------
echo "===================================="
echo "JOBID        = ${JOBID}"
echo "BKG_NAME     = ${BKG_NAME}"
echo "ISEED        = ${ISEED}"
echo "===================================="

# -----------------------------
# Prepare launch card
# -----------------------------
LAUNCH_CARD="${BKG_NAME}_iseed${ISEED}.madevent"
cp background.madevent "${LAUNCH_CARD}"

sed -i \
    -e "s|__ISEED__|${ISEED}|g" \
    "${LAUNCH_CARD}"
echo "Prepared launch card: ${LAUNCH_CARD}"

# -----------------------------
# Untar production directory
# -----------------------------
echo "# Untar the Production Dir"
tar -xaf "${BKG_NAME}.tar.gz"
echo "Done"

# -----------------------------
# Run MadEvent
# -----------------------------
./${BKG_NAME}/bin/madevent ${LAUNCH_CARD}
wait

cp ${BKG_NAME}/Events/run_01/tag_1_delphes_events.root ${BKG_NAME}_${ISEED}_delphes_events.root

if [ -e "${BKG_NAME}_${ISEED}_delphes_events.root" ]; then
  until xrdcp -f ${BKG_NAME}_${ISEED}_delphes_events.root root://eoscms.cern.ch//eos/cms/store/group/phys_susy/sus-23-008/DelphesBackgroundFiles/${BKG_NAME}_${ISEED}_delphes_events.root; do
    sleep 60
    echo "Retrying"
  done
fi
exitcode=$?

if [ ! -e "${BKG_NAME}_${ISEED}_delphes_events.root" ]; then
  echo "Error: The python script failed, could not create the output file."
fi
exit $exitcode

echo "===================================="
echo "Job finished successfully"
echo "===================================="
