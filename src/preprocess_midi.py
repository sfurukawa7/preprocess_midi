import pretty_midi as pm
import mido
import numpy as np
import math
import glob
from pathlib import Path
from tqdm import tqdm

from sklearn.model_selection import train_test_split

print_anything = True 


        
def piano_roll_to_pretty_midi(piano_roll, highest_note, lowest_note, fs, tempo, has_cropped=False, program=0):
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.
    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.
    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.
    '''
    if has_cropped:
        piano_roll = np.pad(piano_roll, ((lowest_note, 128-highest_note), (0,0)), 'constant', constant_values = (0,0)) # cropした分を戻す

    notes, frames = piano_roll.shape
    mid = pm.PrettyMIDI(initial_tempo=tempo)
    instrument = pm.Instrument(program=program)
    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pm.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    mid.instruments.append(instrument)
    return mid 

#where to cut off high notes. 84 = C6
#where to cut off low notes. 24=C1
def load_rolls(file, smallest_note = 48, maximal_number_of_voices_per_track = 1, extracted_bar_num = 2, highest_note=84, lowest_note=24, notes_count_threshold = 0.5, max_velocity = 127, save_reconstructed_midi=False):
    print(file)
    # MIDIをロード
    # 失敗した場合はNoneを返す
    try:
        mid = pm.PrettyMIDI(file)
    except (ValueError, EOFError, IndexError, OSError, KeyError, ZeroDivisionError, AttributeError) as e:
        exception_str = 'Unexpected error in ' + file + ':\n', e, sys.exc_info()[0]
        if print_anything:print(exception_str)
        return None, None, None, None, None, None

    if print_anything:print("Time signature changes: ", mid.time_signature_changes)
    # 曲の始まりと終わりを決定
    # 途中でテンポが変わる場合は、テンポが安定している最も長い箇所(以降、「選択箇所」と呼びます)を選択
    # サイレント部分はカット
    # 小節の開始地点が曲中で一致しているか確認する
    time_signature_list = mid.time_signature_changes
    tempo_change_times, tempo_change_bpm = mid.get_tempo_changes()

    song_start = 0
    song_end = mid.get_end_time()

    # 4/4拍子の区間を抽出
    time_signature_terms = [] # 区間辞書を格納するリスト
    for i, ts in enumerate(time_signature_list):
        if ts.numerator == 4 and ts.denominator == 4:
            time_signature_term = {}
            time_signature_term['start_time'] = ts.time
            time_signature_term['end_time'] = time_signature_list[i+1].time if i != len(time_signature_list)-1 else song_end

            time_signature_terms.append(time_signature_term)

    # # 拍子記号の調整
    # if len(time_signature_list) > 1:
    #     longest_part = 0
    #     longest_part_start_time = 0
    #     longest_part_end_time = song_end
    #     longest_part_tempo = 0

    #     time_signature_change_times = np.array([ts.time for ts in time_signature_list])

    #     _time_signature_change_times = np.append(time_signature_change_times, song_end) # tempo_change_timesに音楽の終了時刻を追加
    #     time_signature_interval_list = np.diff(_time_signature_change_times, n=1) # 各テンポの演奏時間のリストを生成
    #     longest_part_index = time_signature_interval_list.argmax() # 最長のテンポのインデックスを取得
        
    #     longest_part = time_signature_interval_list.max() # 最長のテンポの時間を取得
    #     longest_part_start_time = _time_signature_change_times[longest_part_index]
    #     longest_part_end_time = _time_signature_change_times[longest_part_index+1]
    #     longest_part_time_signature = time_signature_list[longest_part_index]

    #     song_start = longest_part_start_time
    #     song_end = longest_part_end_time
    #     longest_part_time_signature.time = 0.0 # 指定した拍子が時刻0からになるように設定
    #     time_signature = longest_part_time_signature
    # else:
    #     time_signature = time_signature_list[0]

    # テンポの調整
    if len(tempo_change_times) > 1:
        longest_part = 0
        longest_part_start_time = 0
        longest_part_end_time = song_end
        longest_part_tempo = 0
        
        _tempo_change_times = np.append(tempo_change_times, song_end) # tempo_change_timesに音楽の終了時刻を追加
        tempo_interval_list = np.diff(_tempo_change_times, n=1) # 各テンポの演奏時間のリストを生成
        longest_part_index = tempo_interval_list.argmax() # 最長のテンポのインデックスを取得
        
        longest_part = tempo_interval_list.max() # 最長のテンポの時間を取得
        longest_part_start_time = _tempo_change_times[longest_part_index]
        longest_part_end_time = _tempo_change_times[longest_part_index+1]
        longest_part_tempo = tempo_change_bpm[longest_part_index]

        song_start = longest_part_start_time
        song_end = longest_part_end_time
        tempo = longest_part_tempo
    else:
        tempo = tempo_change_bpm[0]

    # piano_mid = pm.PrettyMIDI(initial_tempo=tempo)
    # piano_mid.time_signature_changes = [time_signature]

    # # ピアノのみ抽出 
    # piano_notes_list = []
    # for instrument in mid.instruments:
    #     print(instrument)
    #     if instrument.name == 'Piano': # ピアノのみ実行
    #     # if instrument.program == 0 and instrument.is_drum == False: # ピアノのみ実行
    #         new_notes = [] # ピアノ音を入れるリスト
    #         for note in instrument.notes:
    #             # 選択箇所区間に入っているかチェック
    #             if note.start >= song_start and note.end <= song_end:
    #                 # 時刻を選択箇所内に揃える
    #                 note.start -= song_start
    #                 note.end -= song_start
    #                 new_notes.append(note)
    #         piano_notes_list.append(new_notes)

    # ピアノのみ抽出 
    piano_instrument_list = []
    for instrument in mid.instruments:
        if instrument.program == 0 and instrument.is_drum == False: # ピアノのみ実行
            piano_instrument_list.append(instrument)

    if len(piano_instrument_list) > 0:
        number_of_notes = []
        piano_rolls = [i.get_piano_roll(fs=100) for i in piano_instrument_list]
        for piano_roll in piano_rolls:
            number_of_notes.append(np.count_nonzero(piano_roll))
        
        np_number_of_notes = np.array(number_of_notes)
        chosen_piano_index = np_number_of_notes.argmax()
        piano_instrument = piano_instrument_list[chosen_piano_index]

        new_notes = [] # ピアノ音を入れるリスト
        for note in piano_instrument.notes:
            # 選択箇所区間に入っているかチェック
            if note.start >= song_start and note.end <= song_end:
                # 時刻を選択箇所内に揃える
                note.start -= song_start
                note.end -= song_start
                new_notes.append(note)

        piano_instrument.notes = new_notes
        print(piano_instrument)
    else:
        return None

    quarter_note_length = 60. / tempo
    # fs：サンプリング周波数
    # quarter_note_length * 4.：1小節の長さ(s)
    # smallest_noteは最小音符(e.g. 16分音符ならsmallest_note = 16)
    fs = 1./(quarter_note_length * 4./ smallest_note)
    total_ticks = math.ceil(song_end*fs)

    instrument = mid.instruments[0]
    # np.ndarrayでpiano_rollを作成
    piano_roll = np.zeros((total_ticks, 128))
    for note in instrument.notes:
        note_tick_start = note.start * fs
        note_tick_end = note.end * fs
        absolute_start = int(round(note_tick_start))
        absolute_end = int(round(note_tick_end))
        decimal = note_tick_start -absolute_start

        # tick付近の音or1s以上の音ならばpiano_rollに追加(このため実際の音楽と少し異なる)
        if decimal < 10e-3 or absolute_end-absolute_start >= 1:
            piano_roll[absolute_start:absolute_end, note.pitch] = note.velocity

    cropped_piano_roll = piano_roll[:, lowest_note:highest_note]

    extracted_bars_ticks = extracted_bar_num * smallest_note
    total_ticks = cropped_piano_roll.shape[0]
    if total_ticks % extracted_bars_ticks == 0:
        total_extracted_bars_num = total_ticks / extracted_bars_ticks        
        splited_piano_rolls = np.split(cropped_piano_roll, total_extracted_bars_num)
    else:
        padding_length = extracted_bars_ticks - (total_ticks % extracted_bars_ticks)
        padded_piano_roll = np.pad(cropped_piano_roll, ((0, padding_length), (0,0)), 'constant', constant_values = (0,0)) # padding_length分だけパディング
        padded_total_ticks = padded_piano_roll.shape[0]
        total_extracted_bars_num = padded_total_ticks / extracted_bars_ticks        
        splited_piano_rolls = np.split(padded_piano_roll, total_extracted_bars_num)

    # 休符の多い小節を除去
    new_piano_rolls = []
    for splited_piano_roll in splited_piano_rolls:
        notes_or_rests_list = [1 if one_tick_piano_roll.sum() > 0 else 0 for one_tick_piano_roll in splited_piano_roll]
        notes_percent = sum(notes_or_rests_list) / len(splited_piano_roll)

        # 音数の閾値を超えたピアノロールのみappend
        if notes_percent > notes_count_threshold:
            new_piano_rolls.append(splited_piano_roll)

    if save_reconstructed_midi:
        pathlib_file = Path(file)
        file_stem = pathlib_file.stem
        style_name = pathlib_file.parent.name

        # 保存用ディレクトリを作成
        reconstructed_data_dir = Path('./reconstructed_data') / style_name
        if not reconstructed_data_dir.is_dir():
            reconstructed_data_dir.mkdir(parents=True)
        
        for i, new_piano_roll in enumerate(new_piano_rolls):
            reconstructed_mid = piano_roll_to_pretty_midi(new_piano_roll.T, highest_note, lowest_note, fs, tempo, has_cropped=True)
            reconstructed_file = reconstructed_data_dir / (file_stem + '_{}.mid'.format(i))
            reconstructed_mid.write(str(reconstructed_file))

    np_new_piano_rolls = np.array(new_piano_rolls)
    X = np.where(np_new_piano_rolls > 0, 1, 0) # 学習用ピアノロール
    return X
    # # 各楽器のprogram numberを入手 
    # # program number:0~127の楽器と一対一で対応する番号
    # programs = [i.program for i in mid.instruments]

    # if print_anything:print(max_concurrent_notes_per_track_list)
    # # 事前に設定した最大発音数を超えないようにするためのリスト
    # override_max_notes_per_track_list = [maximal_number_of_voices_per_track for _ in max_concurrent_notes_per_track_list] 
    # # トラック数をmax_voice個に制限したときに生じる無音のトラック数(そもそも元々のトラック数がmax_voice未満の場合、無音のトラックが生じる)
    # silent_tracks_if_we_dont_override = config.MAX_VOICES - sum([min(maximal_number_of_voices_per_track, x) if x > 0 else 0 for x in max_concurrent_notes_per_track_list[:config.MAX_VOICES]])

    # if print_anything:print("Silent tracks if no override:", silent_tracks_if_we_dont_override)
    # # 「silent_tracks_if_we_dont_overrideが存在andあるトラックの最大発音数が設定よりオーバーしている」とき
    # # ->silent_tracks_if_we_dont_override分だけオーバーしているトラックの最大発音数の上限を上げることができる
    # for voice in range(min(config.MAX_VOICES, len(max_concurrent_notes_per_track_list))):
    #     if silent_tracks_if_we_dont_override > 0 and max_concurrent_notes_per_track_list[voice] > maximal_number_of_voices_per_track:
    #         additional_voices = min(silent_tracks_if_we_dont_override, max_concurrent_notes_per_track_list[voice]-maximal_number_of_voices_per_track)
    #         override_max_notes_per_track_list[voice] += additional_voices
    #         silent_tracks_if_we_dont_override -= additional_voices
    # if print_anything:print("Override programs:", override_max_notes_per_track_list)

    # # 発音数を最大同時発音数以内にするためにpiano_rollを選択する
    # chosen_piano_rolls = []
    # chosen_velocity_rolls = []
    # chosen_held_note_rolls = []
    # chosen_programs = []
    # max_song_length = 0

    # for piano_roll, velocity_roll, held_note_roll, program, max_concurrent_notes, override_max_notes_per_track in zip(piano_rolls, velocity_rolls, held_note_rolls, programs, max_concurrent_notes_per_track_list, override_max_notes_per_track_list):
    #     if max_concurrent_notes > 0:
    #         if include_only_monophonic_instruments:
    #             if max_concurrent_notes > 1:
    #                 if print_anything:print("Skipping this piano roll since it's polyphonic. Prigram number ", program)
    #                 continue
    #             else:
    #                 pass
    #                 if print_anything:print("Adding monophonic program number: ", program)
                
    #             monophonic_piano_roll = piano_roll

    #             if len(chosen_piano_rolls) < config.MAX_VOICES:
    #                 chosen_piano_rolls.append(monophonic_piano_roll)
    #                 chosen_velocity_rolls.append(velocity_roll)
    #                 chosen_held_note_rolls.append()
    #                 chosen_programs.append(program)
    #                 if monophonic_piano_roll.shape[0] > max_song_length:
    #                     max_song_length = monophonic_piano_roll.shape[0]
    #                 else:
    #                     break
    #         else: # 初期設定ではこちら
    #             # トラックあたりのボイス数を、トラックあたりの実際の同時使用ボイスの最小値、または設定ファイルで許可されている最大値で制限
    #             for voice in range(min(max_concurrent_notes, max(maximal_number_of_voices_per_track, override_max_notes_per_track))):
    #                 # 最高音をvoice 0, その次をvoice 1, ...とする
    #                 monophonic_piano_roll = np.zeros(piano_roll.shape)
    #                 for step in range(piano_roll.shape[0]):
    #                     # 音の高い順に並び替え
    #                     notes = np.nonzero(piano_roll[step, :])[0][::-1]
    #                     if len(notes) > voice:
    #                         monophonic_piano_roll[step, notes[voice]] = 1
                    
    #                 # これらをchosenに追加する
    #                 if len(chosen_piano_rolls) < config.MAX_VOICES:
    #                     chosen_piano_rolls.append(monophonic_piano_roll)
    #                     chosen_velocity_rolls.append(velocity_roll[:, voice])
    #                     chosen_held_note_rolls.append(held_note_roll[:, voice])
    #                     chosen_programs.append(program)
    #                     if monophonic_piano_roll.shape[0] > max_song_length:
    #                         max_song_length = monophonic_piano_roll.shape[0]
    #                 else:
    #                     break
    #             if len(chosen_piano_rolls) == config.MAX_VOICES:
    #                 break
    
    # assert(len(chosen_piano_rolls) == len(chosen_velocity_rolls))
    # assert(len(chosen_piano_rolls) == len(chosen_held_note_rolls))
    # assert(len(chosen_piano_rolls) == len(chosen_programs))

    # if len(chosen_piano_rolls) > 0:
    #     song_length = max_song_length*config.MAX_VOICES

    #     # Yの準備
    #     # Y:ターゲットノート
    #     Y = np.zeros((song_length, chosen_piano_rolls[0].shape[1]))
    #     # テンソルとなっているピアノロールを展開して一つのマトリックスに
    #     for i, piano_roll in enumerate(chosen_piano_rolls):
    #         for step in range(piano_roll.shape[0]):
    #             Y[i + step*config.MAX_VOICES, :] += piano_roll[step, :]
        
    #     for step in range(Y.shape[0]):
    #         # 演奏される音は最大でも1音
    #         assert(np.sum(Y[step, :]) <= 1)
    #     # 想定する音高の範囲で切り取る
    #     # これにより特徴空間を大幅に削減できる
    #     Y = Y[:, config.LOW_CROP:config.HIGH_CROP]
    #     # 無音を追加
    #     if include_silent_note:
    #         Y = np.append(Y, np.zeros((Y.shape[0], 1)), axis=1)
    #         for step in range(Y.shape[0]):
    #             if np.sum(Y[step]) == 0:
    #                 Y[step, -1] = 1
    #         # 各ステップのどこかには1があるようにする
    #         for step in range(Y.shape[0]):
    #             assert(np.sum(Y[step, :] == 1))

def import_midi_from_folder(folder, load_from_npy_instead_of_midi, save_imported_midi_as_npy, save_reconstructed_midi, test_fraction):
    # npyでロードする場合
    if load_from_npy_instead_of_midi:
        print("X: Now Loading ...")
        X_train = np.load(str(Path(folder) / 'X_train.npy'), allow_pickle=True)
        X_test = np.load(str(Path(folder) / 'X_test.npy'), allow_pickle=True)
        
        print("c: Now Loading ...")
        c_train = np.load(str(Path(folder) / 'c_train.npy'), allow_pickle=True)
        c_test = np.load(str(Path(folder) / 'c_test.npy'), allow_pickle=True)
        
        print("paths: Now Loading ...")
        train_paths = np.load(str(Path(folder) / 'train_paths.npy'), allow_pickle=True)
        test_paths = np.load(str(Path(folder) / 'test_paths.npy'), allow_pickle=True)

        return X_train, X_test, c_train, c_test, train_paths, test_paths

    X_list = []
    c_classes = []
    paths = []

    no_imported = 0

    folder_path = Path(folder)
    class_dirs = list([p for p in folder_path.iterdir() if p.is_dir()])
    class_names = list([p.name for p in folder_path.iterdir() if p.is_dir()])
    if print_anything:print(class_names)

    for i, class_dir in enumerate(class_dirs):
        print("class {}: Now loading ...".format(class_names[i]))
        files = list([p for p in class_dir.glob('**/*.mid')])

        for file in tqdm(files):
            if file.suffix == '.mid' or file.suffix == '.midi':
                try:
                    X = load_rolls(str(file), save_reconstructed_midi=save_reconstructed_midi)
                except mido.midifiles.meta.KeySignatureError as e:
                    print(str(file), e)
                    X = None
                C = i
                        
                if X is not None:
                    X_list.append(X)
                    c_classes.append(C)
                    paths.append(str(file))
                    no_imported += 1

        # if no_imported >= config.MAX_SONGS:
        #     break
    
    if print_anything:print(len(X_list))
    assert(len(X_list) == len(paths))
    assert(len(X_list) == len(c_classes))

    # unique, counts = np.unique(c_classes, return_counts=True)
    # if print_anything:print(dict(zip(unique, counts)))

    X_train, X_test, c_train, c_test, train_paths, test_paths = train_test_split(X_list, c_classes, paths, test_size=test_fraction, random_state=42, stratify=c_classes)

    # if print_anything:print(c_train)

    # 各スタイルの入力数が同じになるようにする
    equal_mini_songs = True
    if equal_mini_songs:
        splits_per_class = np.zeros((len(class_names),))
        for i, song in enumerate(X_train):
            # クラスのインデックスを取得
            c = c_train[i]
            # 入力数を追加
            splits_per_class[c] += len(song)

        amount_of_splits = min(splits_per_class)
        if print_anything:print(splits_per_class)
        if print_anything:print(amount_of_splits)

        # amount_of_splitsを超えないように各スタイルの曲数を調節して、最終的に使用するデータセットを作成する
        X_train_new = []
        c_train_new = []
        train_paths_new = []
        splits_per_class_new = np.zeros((len(class_names),))
        for i, song in enumerate(X_train):
            c = c_train[i]
            if splits_per_class_new[c] + len(song) <= amount_of_splits:
                X_train_new.append(X_train[i])
                c_train_new.append(c_train[i])
                train_paths_new.append(train_paths[i])
                splits_per_class_new[c] += len(song)

        if print_anything:print(splits_per_class_new)
        X_train = X_train_new
        c_train = c_train_new
        train_paths = train_paths_new

    if save_imported_midi_as_npy:
        # 保存用ディレクトリを作成
        npy_dataset_dir = Path('./npy') 
        if not npy_dataset_dir.is_dir():
            npy_dataset_dir.mkdir(parents=True)

        np.save(str(npy_dataset_dir / 'X_train.npy') , X_train)
        np.save(str(npy_dataset_dir / 'X_test.npy'), X_test)
        
        np.save(str(npy_dataset_dir / 'c_train.npy'), c_train)
        np.save(str(npy_dataset_dir / 'c_test.npy'), c_test)
        
        np.save(str(npy_dataset_dir / 'train_paths.npy'), train_paths)
        np.save(str(npy_dataset_dir / 'test_paths.npy'), test_paths)

    return X_train, X_test, c_train, c_test, train_paths, test_paths

def check_time_signature(path):
    # MIDIをロード
    # 失敗した場合はNoneを返す
    try:
        mid = pm.PrettyMIDI(path)
    except (ValueError, EOFError, IndexError, OSError, KeyError, ZeroDivisionError, AttributeError) as e:
        exception_str = 'Unexpected error in ' + path + ':\n', e, sys.exc_info()[0]
        if print_anything:print(exception_str)
        return None, None, None, None, None, None

    tempo_change_times, tempo_change_bpm = mid.get_tempo_changes()
    print(tempo_change_bpm)
    # if print_anything:print("Time signature changes: ", mid.time_signature_changes)

if __name__ == '__main__':
    folder = '../data'
    # folder = './npy'
    import_midi_from_folder(folder, False, True, True, 1/3)
    # path = '../data/sample/jazz_Chipblue.mid'
    # load_rolls(path, smallest_note = 48, maximal_number_of_voices_per_track = 1) 

    # path_list = glob.glob('../data/jazz/*.mid')
    # for m in path_list:
    #     print(m)
    #     check_time_signature(m)