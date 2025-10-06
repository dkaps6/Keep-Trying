name: Run pipeline

on:
  workflow_dispatch:
    inputs:
      date:
        description: "Logical date for the run (YYYY-MM-DD or 'today')"
        required: true
        default: "today"
      season:
        description: "Season tag (e.g., 2025)"
        required: true
        default: "2025"
      window_hours:
        description: "Only price events starting within N hours (0 = no filter)"
        required: true
        default: "36"
      cap:
        description: "Hard cap on number of events to fetch (0 = no cap)"
        required: true
        default: "0"
      books:
        description: "Bookmakers (comma separated keys)"
        required: true
        default: "draftkings,fanduel,betmgm,caesars"
      markets:
        description: "Override markets (comma separated). Leave blank for defaults."
        required: false
        default: ""
      order:
        description: "Provider sorting (usually 'odds')"
        required: false
        default: "odds"
      teams:
        description: "Only include games where team name contains any of these (comma separated). Leave blank for full slate."
        required: false
        default: ""
      selection:
        description: "Optional selection filter (regex/substring on player name). Leave blank for none."
        required: false
        default: ""

jobs:
  run:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    env:
      # Odds API key comes from repo secrets
      THE_ODDS_API_KEY: ${{ secrets.THE_ODDS_API_KEY }}

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

      - name: Run model
        env:
          DATE:       ${{ github.event.inputs.date }}
          SEASON:     ${{ github.event.inputs.season }}
          WINDOW:     ${{ github.event.inputs.window_hours }}
          CAP:        ${{ github.event.inputs.cap }}
          BOOKS:      ${{ github.event.inputs.books }}
          MARKETS:    ${{ github.event.inputs.markets }}
          ORDER:      ${{ github.event.inputs.order }}
          TEAMS:      ${{ github.event.inputs.teams }}
          SELECTION:  ${{ github.event.inputs.selection }}
        run: |
          set -euo pipefail

          ARGS="--date \"$DATE\" --season \"$SEASON\" --window \"$WINDOW\" --cap \"$CAP\" --books \"$BOOKS\""

          # Only include optional flags if non-empty to avoid argparse errors
          if [ -n "$MARKETS" ];   then ARGS="$ARGS --markets \"$MARKETS\""; fi
          if [ -n "$ORDER" ];     then ARGS="$ARGS --order \"$ORDER\""; fi
          if [ -n "$TEAMS" ];     then ARGS="$ARGS --teams \"$TEAMS\""; fi
          if [ -n "$SELECTION" ]; then ARGS="$ARGS --selection \"$SELECTION\""; fi

          echo "Running: python run_model.py $ARGS"
          python run_model.py $ARGS

      - name: Upload outputs
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: model-outputs
          path: |
            outputs/**
          if-no-files-found: warn


if __name__ == "__main__":
    sys.exit(main())
